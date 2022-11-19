from config import config_reader

def main():
    print(config_reader.read_config("./config.json"))

main()