from datetime import datetime


class ModelSettings:
    """
    Object that is passed through functions with information like paths of input data and config and on parts of model
    should be saved and where.
    """
    def __init__(self, data_path, config_path, save_outputs, saved_model_path=None, test_mode=False):
        self.data_path = data_path
        self.config_path = config_path

        now = datetime.now()
        datetime_of_run = now.strftime("%m-%d-%y_%H:%M:%S")

        self.save_outputs = save_outputs
        if save_outputs:
            self.output_ast_path = 'output/' + datetime_of_run + '/asts/'
            self.output_data_path = 'output/' + datetime_of_run + '/data/'
            self.output_model_path = 'output/' + datetime_of_run + '/models/'
            self.output_plots_path = 'output/' + datetime_of_run + '/plots/'
            self.output_config_path = 'output/' + datetime_of_run + '/configs/'

        if saved_model_path is not None:
            self.saved_model_path = saved_model_path

        self.test_mode = test_mode
