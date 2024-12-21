import logging
from configuration_loader import ConfigurationLoader
from run_strategy import RunStrategy

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

if __name__ == "__main__":
    CONFIG_DIR = "configs/"  # Directory containing JSON configuration files
    SCHEMA_DIR = "schemas/"  # Directory containing JSON schemas

    # Load configurations
    loader = ConfigurationLoader(CONFIG_DIR, SCHEMA_DIR)
    configs = loader.load_and_validate_configs()

    # Run strategies
    if configs:
        strategy = RunStrategy(configs)
        strategy.run()
    else:
        logger.error("No valid configurations found. Exiting.")
