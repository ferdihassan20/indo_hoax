import argparse

def input_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__' :

    config = input_parser()

    print(config.device)