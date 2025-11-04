from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        print("🚀 Stage_03_model_training.py started running!")
    try:
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)

        print("✅ Configuration loaded successfully.")
        training.get_base_model()
        print("✅ Base model loaded successfully.")
        training.train_valid_generator()
        print("✅ Data generators created successfully.")
        training.train()
        print("✅ Model training finished successfully.")
    except Exception as e:
        import traceback
        print("❌ ERROR in stage_03_model_training.py:", e)
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Stage_03_model_training.py script executed directly!")
    obj = ModelTrainingPipeline()
    obj.main()
