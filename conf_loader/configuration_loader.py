import os
import json
import logging
from typing import List, Dict
from jsonschema import validate, ValidationError

logger = logging.getLogger("ConfigurationLoader")


class ConfigurationLoader:
    """
    Responsible for loading and validating configurations from a directory.
    """
    def __init__(self, config_dir: str, schema_dir: str):
        self.config_dir = config_dir
        self.schema_dir = schema_dir

    def _load_schema(self, schema_name: str) -> Dict:
        """Load a JSON schema by name."""
        schema_path = os.path.join(self.schema_dir, schema_name)
        with open(schema_path, 'r') as schema_file:
            return json.load(schema_file)

    def load_and_validate_configs(self) -> List[Dict]:
        """
        Load all JSON configurations from the config directory and validate them.

        Returns:
            List[Dict]: List of validated configurations.
        """
        configs = []
        for config_file in os.listdir(self.config_dir):
            if config_file.endswith(".json"):
                config_path = os.path.join(self.config_dir, config_file)
                with open(config_path, 'r') as file:
                    config_data = json.load(file)
                try:
                    schema_name = "detection_schema.json" if "ul_detection" in config_data else "segmentation_schema.json"
                    schema = self._load_schema(schema_name)
                    validate(instance=config_data, schema=schema)
                    configs.append(config_data)
                except ValidationError as e:
                    logger.error(f"Validation failed for {config_file}: {e}")
                except Exception as e:
                    logger.error(f"Error loading {config_file}: {e}")
        return configs
