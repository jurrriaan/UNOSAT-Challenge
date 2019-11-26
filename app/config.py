import configparser

class Config:
    def __init__(self, config_path='config.ini'):
        self._config_path=config_path
        self._config = {}

    def config(self, section='default'):
        """Reads the config.ini file and returns the configuration.

        By default reads the default section

        Args:
            section (string): Section to read from the config.ini

        Returns:
            config (config.Sections)
        """
        if self._config:
            return self._config[section]
        
        config = configparser.ConfigParser()
        config.read(self._config_path)
        self._config = config
        return self._config[section]