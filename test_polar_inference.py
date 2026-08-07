#!/usr/bin/env python3
"""
Тесты для polar_inference.py — поведенческие твики.
Запуск: pytest test_polar_inference.py -v
Время: ~2 секунды (без реальных моделей).
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Импорты тестируемых сущностей
from polar_inference import (
    ModelRouter, ModelManager,
    mode_single, mode_a_parallel, mode_b_router,
    looks_like_refusal,
    GEN_PARAMS, LAPLACE, ABSTAIN_THRESHOLD,
)


# ============================================================
# Фикстуры
# ============================================================
@pytest.fixture
def router():
    return ModelRouter()


@pytest.fixture
def mock_manager():
    """Мок ModelManager: возвращает предсказуемые тексты."""
    m = Mock(spec=ModelManager)

    def fake_generate(model_name, prompt, system_role="",
                      max_new_tokens=None, repetition_penalty=None):
        # Записываем параметры для проверки
        if not hasattr(m, 'call_log'):
            m.call_log = []
        m.call_log.append({
            'model': model_name,
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': repetition_penalty,
        })
        # Ответ зависит от модели (чтобы отличать)
        text = f"[{model_name}] response to: {prompt[:30]}"
        return text, 0.5

    m.generate.side_effect = fake_generate
    return m


@pytest.fixture
def mock_manager_with_refusal(mock_manager):
    """Мок, где instruct выдаёт отказ, coder — нормальный ответ."""
    def fake_generate(model_name, prompt, **kwargs):
        if not hasattr(mock_manager, 'call_log'):
            mock_manager.call_log = []
        mock_manager.call_log.append({'model': model_name, **kwargs})
        if model_name == 'instruct':
            return "I cannot help with that. As an AI...", 0.5
        return "Here is the code...", 0.5
    mock_manager.generate.side_effect = fake_generate
    return mock_manager


# ============================================================
# 1. Лаплас-калибровка confidence
# ============================================================
class TestLaplaceCalibration:
    """(winner + 1) / (total + 2)"""

    def test_empty_prompt_gives_half(self, router):
        d = router.route("")
        assert d.confidence == 0.5

    def test_single_weak_signal(self, router):
        # "python" → code=1, general=0 → (1+1)/(1+2)=0.667
        d = router.route("python")
        assert d.model == 'coder'
        assert abs(d.confidence - 2/3) < 0.01

    def test_strong_code_signal(self, router):
        # "write a python function" → task verb (3) + weak (1) = 4
        # vs 0 → (4+1)/(4+2) = 5/6 ≈ 0.833
        d = router.route("write a python function")
        assert d.model == 'coder'
        assert abs(d.confidence - 5/6) < 0.01

    def test_tie_nonzero(self, router):
        # "explain python" → explain (+1 general) vs python (+1 code)
        # code_score == general_score → confidence = 0.5
        d = router.route("explain python")
        assert d.code_score == d.general_score
        assert d.confidence == 0.5

    def test_moderate_confidence(self, router):
        # "what is wrong with this code" → task verb coder (3) + "what is" general (1)
        # code=3, general=1 → (3+1)/(4+2) = 2/3 ≈ 0.667
        d = router.route("what is wrong with this code")
        assert d.model == 'coder'
        assert abs(d.confidence - 2/3) < 0.01


# ============================================================
# 2. Убранный return-паттерн не срабатывает на прозе
# ============================================================
class TestReturnPatternFix:
    def test_return_in_prose_not_counted(self, router):
        # Обычный английский текст не должен давать code_score
        d = router.route("return the book to the library")
        assert d.code_score == 0
        assert d.model == 'instruct'  # default при 0-0


# ============================================================
# 3. looks_like_refusal
# ============================================================
class TestRefusalDetection:
    @pytest.mark.parametrize("text", [
        "I cannot help with that request.",
        "I can't do that.",
        "As an AI, I'm unable to...",
        "I am not able to assist.",
        "I apologise, but I cannot.",
    ])
    def test_detects_refusal(self, text):
        assert looks_like_refusal(text) is True

    @pytest.mark.parametrize("text", [
        "Here is the code:",
        "A segmentation fault is...",
        "def hello():\n    print('hello')",
        "I can help you with that.",  # "I can" не должно ловиться!
    ])
    def test_ignores_normal(self, text):
        assert looks_like_refusal(text) is False


# ============================================================
# 4. Абстенция по uncertainty (низкая confidence)
# ============================================================
class TestAbstentionUncertainty:
    def test_low_confidence_triggers_both(self, mock_manager):
        # Промпт без явных сигналов — должна быть абстенция
        # "hello" → 0 vs 0 → 0.5 < 0.6 → uncertainty
        result = mode_b_router(mock_manager, "hello")
        assert result['abstained'] is True
        assert 'uncertainty' in result['mode']
        # Обе модели должны быть вызваны
        models_called = [c['model'] for c in mock_manager.call_log]
        assert 'instruct' in models_called
        assert 'coder' in models_called


# ============================================================
# 5. Абстенция по tie (равные ненулевые сигналы)
# ============================================================
class TestAbstentionTie:
    def test_tie_triggers_both(self, mock_manager):
        # "explain python" → explain (1 general) vs python (1 code)
        result = mode_b_router(mock_manager, "explain python")
        assert result['abstained'] is True
        assert 'tie' in result['mode']
        models_called = [c['model'] for c in mock_manager.call_log]
        assert set(models_called) == {'instruct', 'coder'}


# ============================================================
# 6. Абстенция по ambiguity (смешанная задача)
# ============================================================
class TestAbstentionAmbiguity:
    @pytest.mark.parametrize("prompt", [
        "Write a Python function, then explain how it works",
        "Implement the algorithm and also describe it",
        "Translate this code to French",
        "Fix the bug additionally document the changes",
    ])
    def test_ambiguous_prompts_trigger_both(self, mock_manager, prompt):
        result = mode_b_router(mock_manager, prompt)
        assert result['abstained'] is True
        assert 'ambiguity' in result['mode']


# ============================================================
# 7. Уверенный выбор одной модели
# ============================================================
class TestConfidentRouting:
    def test_code_prompt_goes_to_coder(self, mock_manager):
        result = mode_b_router(
            mock_manager, "write a python function to check palindrome")
        assert result['abstained'] is False
        assert result['routing']['model'] == 'coder'
        assert result['mode'] == 'B (Router)'
        assert len(mock_manager.call_log) == 1

    def test_general_prompt_goes_to_instruct(self, mock_manager):
        result = mode_b_router(
            mock_manager, "explain what a neural network is")
        assert result['abstained'] is False
        assert result['routing']['model'] == 'instruct'
        assert len(mock_manager.call_log) == 1


# ============================================================
# 8. Эскалация при отказе выбранной модели
# ============================================================
class TestEscalation:
    def test_refusal_triggers_fallback(self, mock_manager_with_refusal):
        # "translate to French" → instruct (уверенно)
        # но instruct отвечает "I cannot..." → эскалация к coder
        result = mode_b_router(
            mock_manager_with_refusal, "translate to French")
        assert result['abstained'] is True
        assert 'escalation' in result['mode']
        assert result['routing']['escalation_reason'] == 'refusal_signals'
        models_called = [c['model'] for c in mock_manager_with_refusal.call_log]
        assert models_called == ['instruct', 'coder']  # порядок важен


# ============================================================
# 9. Пер-модельные параметры генерации
# ============================================================
class TestPerModelParams:
    def test_coder_uses_its_params(self, mock_manager):
        mode_b_router(mock_manager, "write a python function to sort a list")
        call = next(c for c in mock_manager.call_log if c['model'] == 'coder')
        assert call['repetition_penalty'] == GEN_PARAMS['coder']['repetition_penalty']
        assert call['max_new_tokens'] == GEN_PARAMS['coder']['max_new_tokens']

    def test_instruct_uses_its_params(self, mock_manager):
        mode_b_router(mock_manager, "explain what a neural network is")
        call = next(c for c in mock_manager.call_log if c['model'] == 'instruct')
        assert call['repetition_penalty'] == GEN_PARAMS['instruct']['repetition_penalty']
        assert call['max_new_tokens'] == GEN_PARAMS['instruct']['max_new_tokens']


# ============================================================
# 10. mode_single и mode_a_parallel
# ============================================================
class TestSingleAndParallel:
    def test_mode_single_uses_correct_model(self, mock_manager):
        result = mode_single(mock_manager, "hello", model_name='coder')
        assert result['mode'] == 'Single'
        assert result['model'] == 'coder'
        assert result['kv_caches'] == 1
        assert len(mock_manager.call_log) == 1
        assert mock_manager.call_log[0]['model'] == 'coder'

    def test_mode_parallel_calls_both(self, mock_manager):
        result = mode_a_parallel(mock_manager, "hello")
        assert result['mode'] == 'A (Parallel Twins)'
        assert result['kv_caches'] == 2
        models_called = [c['model'] for c in mock_manager.call_log]
        assert set(models_called) == {'instruct', 'coder'}
        assert 'results' in result
        assert 'instruct' in result['results']
        assert 'coder' in result['results']


# ============================================================
# 11. Lazy loading в ModelManager
# ============================================================
class TestLazyLoading:
    def test_models_not_loaded_at_init(self):
        """При создании ModelManager модели НЕ грузятся сразу."""
        with patch('polar_inference.AutoTokenizer') as mock_tok, \
             patch('polar_inference.AutoModelForCausalLM') as mock_model:
            mock_tok.from_pretrained.return_value = Mock(pad_token=None, eos_token='</s>')
            manager = ModelManager(
                models_config={'instruct': Path('/fake'), 'coder': Path('/fake')},
                weights_files={'instruct': Path('/fake/w'), 'coder': Path('/fake/w')},
                layer_info_files={'instruct': Path('/fake/l'), 'coder': Path('/fake/l')},
            )
            # AutoModelForCausalLM.from_pretrained не должен вызываться
            mock_model.from_pretrained.assert_not_called()
            # Только токенизаторы
            assert mock_tok.from_pretrained.call_count == 2


# ============================================================
# 12. Граничные случаи
# ============================================================
class TestEdgeCases:
    def test_custom_max_tokens_overrides_default(self, mock_manager):
        result = mode_b_router(
            mock_manager, "write a python function",
            max_new_tokens=100)
        call = mock_manager.call_log[0]
        assert call['max_new_tokens'] == 100  # override

    def test_custom_abstain_threshold(self, mock_manager):
        # "python" → conf 0.67; с threshold 0.8 → должна быть абстенция
        result = mode_b_router(
            mock_manager, "python",
            abstain_threshold=0.8)
        assert result['abstained'] is True
        assert 'uncertainty' in result['mode']


# ============================================================
# 13. GPU-пиннинг с TTL и LRU
# ============================================================
class TestGPUPinning:
    def test_autodetect_resident_limit_with_vram(self):
        """Автодетект корректно выбирает лимит."""
        with patch('polar_inference.torch.cuda.is_available', return_value=True), \
             patch('polar_inference.torch.cuda.get_device_properties') as mock_props, \
             patch('polar_inference.AutoTokenizer') as mock_tok:
            # Симулируем 16 ГБ VRAM
            mock_props.return_value.total_memory = 16 * 1024**3
            mock_tok.from_pretrained.return_value = Mock(pad_token=None, eos_token='</s>')

            manager = ModelManager(
                models_config={'instruct': Path('/fake'), 'coder': Path('/fake')},
                weights_files={'instruct': Path('/f'), 'coder': Path('/f')},
                layer_info_files={'instruct': Path('/f'), 'coder': Path('/f')},
            )
            assert manager.max_resident == 2

    def test_second_generate_skips_transfer(self):
        """Второй вызов той же модели не переносит её на GPU снова."""
        manager = self._make_manager_with_mocks(resident_limit=2)

        # Первый вызов: должен перенести на GPU
        manager.generate('instruct', 'hello')
        m = manager.models['instruct']
        assert m.to.call_count == 1
        m.to.reset_mock()
        manager.generate('instruct', 'hello again')
        m.to.assert_not_called()

    def test_lru_eviction_when_limit_exceeded(self):
        """При превышении лимита вытесняется самая старая модель."""
        manager = self._make_manager_with_mocks(resident_limit=1)

        manager.generate('instruct', 'prompt 1')
        manager.generate('coder', 'prompt 2')  # вытесняет instruct

        assert 'coder' in manager.resident_on_gpu
        assert 'instruct' not in manager.resident_on_gpu
        # instruct был выгружен обратно в CPU
        instruct_model = manager.models['instruct']
        cpu_calls = [c for c in instruct_model.to.call_args_list
                     if c.args[0] == 'cpu']
        assert len(cpu_calls) >= 1

    def test_ttl_expires_stale_models(self):
        """Просроченные модели выгружаются при следующем обращении."""
        manager = self._make_manager_with_mocks(resident_limit=2, ttl=1.0)
        manager.generate('instruct', 'hello')

        # Имитируем истечение TTL
        manager.resident_on_gpu['instruct'] = time.time() - 10.0

        # Следующий generate должен выгрузить instruct (TTL)
        manager.generate('coder', 'world')
        assert 'instruct' not in manager.resident_on_gpu

    def test_force_evict(self):
        """Принудительная выгрузка работает."""
        manager = self._make_manager_with_mocks(resident_limit=2)
        manager.generate('instruct', 'hello')
        assert 'instruct' in manager.resident_on_gpu

        manager.force_evict('instruct')
        assert 'instruct' not in manager.resident_on_gpu

    def _make_manager_with_mocks(self, resident_limit, ttl=300.0):
        """Хелпер: ModelManager с моками моделей и токенизаторов."""
        import torch as _torch
        with patch('polar_inference.AutoTokenizer') as mock_tok, \
             patch('polar_inference.torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value.total_memory = 8 * 1024**3

            # Токенизатор-мок: возвращает настоящий dict с тензорами,
            # чтобы .items() и .shape работали. Никаких спец-строк.
            tok = Mock()
            ids = _torch.zeros(1, 5, dtype=_torch.long)
            tok.return_value = {'input_ids': ids, 'attention_mask': ids}
            tok.decode.return_value = "mock response"
            tok.pad_token = None
            tok.eos_token = "mock-eos"
            mock_tok.from_pretrained.return_value = tok

            manager = ModelManager(
                models_config={'instruct': Path('/f'), 'coder': Path('/f')},
                weights_files={'instruct': Path('/f'), 'coder': Path('/f')},
                layer_info_files={'instruct': Path('/f'), 'coder': Path('/f')},
                max_resident_models=resident_limit,
                pin_ttl_seconds=ttl,
            )
            manager.device = 'cpu'  # тесты без GPU

            # Модели-моки: .to() возвращает себя, generate — тензор (1, 10)
            for name in ['instruct', 'coder']:
                m = MagicMock()
                m.to.return_value = m
                m.generate.return_value = _torch.zeros(1, 10, dtype=_torch.long)
                manager.models[name] = m
            return manager

if __name__ == "__main__":
    pytest.main([__file__, "-v"])