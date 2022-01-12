import ruamel.yaml as yaml
import os, sys

class RunningArguments:
    def __init__(self,
        **kwargs
        ):
        self.logging_file = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, 'r') as fp:
            try:
                kwargs = yaml.safe_load(fp)
                return RunningArguments(**kwargs)
            except yaml.YAMLError as e:
                print(e)
                return
    def __str__(self, ):
        output = "\n"
        for k,v in vars(self).items():
            output += f"{str(k)} : {str(v)}\n"
        return output

class TrainingArguments:
    def __init__(self,
        **kwargs
        ):
        self.checkpoint = None
        self.steps_per_epoch = None
        self.start_epoch = 0
        self.start_step = 0
        for k, v in kwargs.items():
            setattr(self, k, v)

    # @staticmethod
    # def load_from_file(file_name):
    #     with open(file_name, 'r') as fp:
    #         try:
    #             kwargs = yaml.safe_load(fp)
    #             return RunningArguments(**kwargs)
    #         except yaml.YAMLError as e:
    #             print(e)
    #             return

    def __str__(self, ):
        output = "\n"
        for k,v in vars(self).items():
            output += f"{str(k)} : {str(v)}\n"
        return output

class DataArguments:
    def __init__(self,
        **kwargs
        ):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, 'r') as fp:
            try:
                kwargs = yaml.safe_load(fp)
                return DataArguments(**kwargs)
            except yaml.YAMLError as e:
                print(e)
                return

    def __str__(self, ):
        output = "\n"
        for k,v in vars(self).items():
            output += f"{str(k)} : {str(v)}\n"
        return output