from padre.experimentcreator import ExperimentCreator
import os


def main():
    experiment_creator = ExperimentCreator()
    print(os.getcwd())
    experiment_creator.parse_config_file('./proof_of_concept/sacred_experiments/cli_config_2.json')
    experiment_creator.execute_experiments()


if __name__ == '__main__':
    main()
