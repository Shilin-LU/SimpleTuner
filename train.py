import logging

# Quiet down, you.
ds_logger1 = logging.getLogger("DeepSpeed")
ds_logger2 = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
ds_logger1.setLevel("ERROR")
ds_logger2.setLevel("ERROR")
import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
from os import environ

environ["ACCELERATE_LOG_LEVEL"] = "WARNING"
environ["CUDA_VISIBLE_DEVICES"] = '5'

from helpers.training.trainer import Trainer
from helpers.training.state_tracker import StateTracker
from helpers import log_format

logger = logging.getLogger("SimpleTuner")
logger.setLevel(environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

import os, json
def load_config(file_path):
    if not os.path.exists(file_path):
        print(f"警告: 配置文件未找到 - {file_path}，将使用默认配置。")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"错误: JSON解码失败 - {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return {}

def convert_dict_to_args(input_dict):
    args_list = []
    for key, value in input_dict.items():
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value == 'true':
                args_list.append(key)
            elif lower_value == 'false':
                # 忽略值为 false 的选项
                continue
            else:
                args_list.append(f"{key}={value}")
        elif isinstance(value, bool):
            if value:
                args_list.append(key)
            else:
                # 忽略值为 False 的选项
                continue
        else:
            args_list.append(f"{key}={value}")
    return args_list

if __name__ == "__main__":
    trainer = None
    try:
        import multiprocessing

        multiprocessing.set_start_method("fork")
    except Exception as e:
        logger.error(
            "Failed to set the multiprocessing start method to 'fork'. Unexpected behaviour such as high memory overhead or poor performance may result."
            f"\nError: {e}"
        )
    try:
        config = load_config('config.json')
        config = convert_dict_to_args(config)
        trainer = Trainer(
            exit_on_error=True,
            config=config,
        )
        trainer.configure_webhook()
        trainer.init_noise_schedule()
        trainer.init_seed()

        trainer.init_huggingface_hub()

        trainer.init_preprocessing_models()
        trainer.init_precision(preprocessing_models_only=True)
        trainer.init_data_backend()
        # trainer.init_validation_prompts()
        trainer.init_unload_text_encoder()
        trainer.init_unload_vae()

        trainer.init_load_base_model()
        trainer.init_controlnet_model()
        trainer.init_precision()
        trainer.init_freeze_models()
        trainer.init_trainable_peft_adapter()
        trainer.init_ema_model()
        # EMA must be quantised if the base model is as well.
        trainer.init_precision(ema_only=True)

        trainer.move_models(destination="accelerator")
        trainer.init_validations()
        trainer.init_benchmark_base_model()

        trainer.resume_and_prepare()

        trainer.init_trackers()
        trainer.train()
    except KeyboardInterrupt:
        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message="Training has been interrupted by user action (lost terminal, or ctrl+C)."
            )
    except Exception as e:
        import traceback

        if StateTracker.get_webhook_handler() is not None:
            StateTracker.get_webhook_handler().send(
                message=f"Training has failed. Please check the logs for more information: {e}"
            )
        print(e)
        print(traceback.format_exc())
    if trainer is not None and trainer.bf is not None:
        trainer.bf.stop_fetching()
