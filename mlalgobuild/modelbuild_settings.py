""" This script contains all settings for all tasks, models and experiments.
"""

import losses
import accuracies

import skopt

# These are settings of the model optimization algorithm
class BaseSettings():
    def __init__(self):
        self.los = losses.LOSSES
        self.acc = accuracies.ACCURACIES
        
        self.search_space = [
            skopt.space.Real(1e-6, 1e-1, "log-uniform", name="learning_rate"),
            skopt.space.Real(0.0, 1.0, "uniform", name="weight_decay"),
            skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_a"),
            skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_b"),
            skopt.space.Real(0.0, 0.9, "uniform", name="dropout"),
            skopt.space.Real(0.0, 1.0, "uniform", name="warmup_len"),
            # skopt.space.Real(0.0, 0.0, "uniform", name="label_smoothing"),
            skopt.space.Integer(1, 100, "log-uniform", name="batch_accum"),
            # skopt.space.Integer(0, 5, "uniform", name="n_head_pow"),
            # skopt.space.Integer(1, 6, "uniform", name="n_layers"),
        ]
        
        self.skopt_kwargs = {
            'n_calls': 30,
            'n_initial_points': 10
        }
        
    def get_criterion(self):
        """ Get criterion - for loss. """
        c1 = self.los['mse']
        c2 = self.los['cos']
        c3 = self.los['cka']
        criterion = lambda pred, gt: c1(pred, gt) + 0.5*c2(pred, gt)
        return criterion
        
    def get_train_summaries(self):
        """ Get training summeries metrics. """
        train_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return train_summaries
        
    def get_dev_summaries(self):
        """ Get dev/validation summeries metrics. """
        dev_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return dev_summaries
        
    def get_scoring(self):
        """ Get scoring function (used to rank models, lower is better). """
        return self.los['mse']
    
    def get_search_space(self):
        """ Get search space of hyperparmeters. """
        return self.search_space
    
    def get_skopt_kwargs(self):
        """ Get skopt additional arguments.
        https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html """
        return self.skopt_kwargs
    
    def get_train_inputs_handler(self):
        """ Get handler function for input batch during training. """
        return lambda x,y: (x,)
    
    def get_train_output_handler(self):
        """ Get handler function for output batch during training. """
        return lambda x,y: y
    
    def get_train_prediction_handler(self):
        """ Get handler function for output batch during training. """
        return lambda p: p
    
    def get_test_prediction_handler(self):
        """ Get handler function for output batch during prediction. """
        return lambda p, dataset: p

        
class Embed2embedMlpMse(BaseSettings):
    def __init__(self):
        super().__init__()
        
    def get_criterion(self):
        """ Get criterion - for loss. """        
        return self.los['mse']
        
    def get_train_summaries(self):
        """ Get training summeries metrics. """
        train_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return train_summaries
        
    def get_dev_summaries(self):
        """ Get dev/validation summeries metrics. """
        dev_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return dev_summaries
        
    def get_scoring(self):
        """ Get scoring function (used to rank models, lower is better). """
        return self.los['mse']
     
        
class Embed2embedMlpMseCos5(Embed2embedMlpMse):
    def __init__(self):
        super().__init__()
        
    def get_criterion(self):
        """ Get criterion - for loss. """        
        c1 = self.los['mse']
        c2 = self.los['cos']
        criterion = lambda pred, gt: c1(pred, gt) + 0.5*c2(pred, gt)
        return criterion

    
class RevdictBaseMse(BaseSettings):
    def __init__(self):
        super().__init__()
        
    def get_criterion(self):
        """ Get criterion - for loss. """        
        return self.los['mse']
        
    def get_train_summaries(self):
        """ Get training summeries metrics. """
        train_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return train_summaries
        
    def get_dev_summaries(self):
        """ Get dev/validation summeries metrics. """
        dev_summaries = {
            'mse': self.los['mse'],
            'cos': self.acc['cos'],
            'cka': self.acc['cka'],
        }
        return dev_summaries
        
    def get_scoring(self):
        """ Get scoring function (used to rank models, lower is better). """
        return self.los['mse']
    
    def get_test_prediction_handler(self):
        """ Get handler function for output batch during prediction. """
        return lambda p, dataset: p.view(-1).cpu().tolist()

    
class DefmodBaseXen(BaseSettings):
    def __init__(self):
        super().__init__()
        
    def get_criterion(self):
        """ Get criterion - for loss. """
        return self.los['xen']
        
    def get_train_summaries(self):
        """ Get training summeries metrics. """
        train_summaries = {
            'xent': self.los['xen'],
            'xent_smooth': self.los['xens'],
            'acc': self.acc['accgls'],
        }
        return train_summaries
        
    def get_dev_summaries(self):
        """ Get dev/validation summeries metrics. """
        dev_summaries = {
            'xent': self.los['xen'],
            'xent_smooth': self.los['xens'],
            'acc': self.acc['accgls'],
        }
        return dev_summaries
        
    def get_scoring(self):
        """ Get scoring function (used to rank models, lower is better). """
        return self.los['xen']
    
    def get_train_inputs_handler(self):
        """ Get handler function for input batch during training. """
        return lambda x,y: (x, y[:-1])
    
    def get_train_output_handler(self):
        """ Get handler function for output batch during training. """
        return lambda x, y: y.view(-1)
    
    def get_train_prediction_handler(self):
        """ Get handler function for output batch during training. """
        return lambda p: p.view(-1, p.size(-1))
    
    def get_test_prediction_handler(self):
        """ Get handler function for output batch during prediction. """
        return lambda p, dataset: dataset.decode(p)

    
class DefmodBaseXens(DefmodBaseXen):
    def __init__(self):
        super().__init__()
    
    def get_criterion(self):
        """ Get criterion - for loss. """
        return self.los['xens']
    
    def get_scoring(self):
        """ Get scoring function (used to rank models, lower is better). """
        return self.los['xens']


    
SETTINGS = {
    # Embed2embed settings
    'embed2embed-mlp-mse': Embed2embedMlpMse, # MSE
    'embed2embed-mlp-mse-cos5': Embed2embedMlpMseCos5, # MSE + 0.5*COS
    # Revdict settings
    'revdict-base-mse': RevdictBaseMse, # MSE
    # Defmod settings
    'defmod-base-xen': DefmodBaseXen, # Cros-entropy
    'defmod-base-xens': DefmodBaseXens, # Cros-entropy smooth labels
}









# """ This script contains all settings for all tasks, models and experiments.
# """

# import losses
# import accuracies

# import skopt

# class BaseSettings():
#     def __init__(self):
#         self.los = losses.LOSSES
#         self.acc = accuracies.ACCURACIES
        
#         self.search_space = [
#             skopt.space.Real(1e-6, 1e-1, "log-uniform", name="learning_rate"),
#             skopt.space.Real(0.0, 1.0, "uniform", name="weight_decay"),
#             skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_a"),
#             skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_b"),
#             skopt.space.Real(0.0, 0.9, "uniform", name="dropout"),
#             skopt.space.Real(0.0, 1.0, "uniform", name="warmup_len"),
#             # skopt.space.Real(0.0, 0.0, "uniform", name="label_smoothing"),
#             skopt.space.Integer(1, 100, "log-uniform", name="batch_accum"),
#             # skopt.space.Integer(0, 5, "uniform", name="n_head_pow"),
#             # skopt.space.Integer(1, 6, "uniform", name="n_layers"),
#         ]
        
#         self.skopt_kwargs = {
#             'n_calls': 30,
#             'n_initial_points': 10
#         }
        
#     def get_criterion(self):
#         """ Get criterion - for loss. """
#         c1 = self.los['mse']
#         c2 = self.los['cos']
#         c3 = self.los['cka']
#         criterion = lambda pred, gt: c1(pred, gt) + 0.5*c2(pred, gt)
#         return criterion
        
#     def get_train_summaries(self):
#         """ Get training summeries metrics. """
#         train_summaries = {
#             'mse': self.los['mse'],
#             'cos': self.acc['cos'],
#             'cka': self.acc['cka'],
#         }
#         return train_summaries
        
#     def get_dev_summaries(self):
#         """ Get dev/validation summeries metrics. """
#         dev_summaries = {
#             'mse': self.los['mse'],
#             'cos': self.acc['cos'],
#             'cka': self.acc['cka'],
#         }
#         return dev_summaries
        
#     def get_scoring(self):
#         """ Get scoring function (used to rank models, lower is better). """
#         return self.los['mse']
    
#     def get_search_space(self):
#         """ Get search space of hyperparmeters. """
#         return self.search_space
    
#     def get_skopt_kwargs(self):
#         """ Get skopt additional arguments.
#         https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html """
#         return self.skopt_kwargs
    
#     def get_inputs_handler(self):
#         """ Get handler function for input batch. """
#         return lambda x,y: (x,)
    
#     def get_output_handler(self):
#         """ Get handler function for output batch. """
#         return lambda x,y: y
    
#     def get_prediction_handler(self):
#         """ Get handler function for output batch. """
#         return lambda p: p

        
# class Embed2embedMlpMse(BaseSettings):
#     def __init__(self):
#         super().__init__()
        
#     def get_criterion(self):
#         """ Get criterion - for loss. """        
#         return self.los['mse']
        
#     def get_train_summaries(self):
#         """ Get training summeries metrics. """
#         train_summaries = {
#             'mse': self.los['mse'],
#             'cos': self.acc['cos'],
#             'cka': self.acc['cka'],
#         }
#         return train_summaries
        
#     def get_dev_summaries(self):
#         """ Get dev/validation summeries metrics. """
#         dev_summaries = {
#             'mse': self.los['mse'],
#             'cos': self.acc['cos'],
#             'cka': self.acc['cka'],
#         }
#         return dev_summaries
        
#     def get_scoring(self):
#         """ Get scoring function (used to rank models, lower is better). """
#         return self.los['mse']
     
        
# class Embed2embedMlpMseCos5(Embed2embedMlpMse):
#     def __init__(self):
#         super().__init__()
        
#     def get_criterion(self):
#         """ Get criterion - for loss. """        
#         c1 = self.los['mse']
#         c2 = self.los['cos']
#         criterion = lambda pred, gt: c1(pred, gt) + 0.5*c2(pred, gt)
#         return criterion

    
# class RevdictBaseMse(BaseSettings):
#     def __init__(self):
#         super().__init__()
        
#     def get_criterion(self):
#         """ Get criterion - for loss. """        
#         return self.los['mse']
        
#     def get_train_summaries(self):
#         """ Get training summeries metrics. """
#         train_summaries = {
#             'mse': self.los['mse'],
#             'cos': self.acc['cos'],
#             'cka': self.acc['cka'],
#         }
#         return train_summaries
        
#     def get_dev_summaries(self):
#         """ Get dev/validation summeries metrics. """
#         dev_summaries = {
#             'mse': self.los['mse'],
#             'cos': self.acc['cos'],
#             'cka': self.acc['cka'],
#         }
#         return dev_summaries
        
#     def get_scoring(self):
#         """ Get scoring function (used to rank models, lower is better). """
#         return self.los['mse']

    
# class DefmodBaseXen(BaseSettings):
#     def __init__(self):
#         super().__init__()
        
#     def get_criterion(self):
#         """ Get criterion - for loss. """
#         return self.los['xen']
        
#     def get_train_summaries(self):
#         """ Get training summeries metrics. """
#         train_summaries = {
#             'xent': self.los['xen'],
#             'xent_smooth': self.los['xens'],
#             'acc': self.acc['accgls'],
#         }
#         return train_summaries
        
#     def get_dev_summaries(self):
#         """ Get dev/validation summeries metrics. """
#         dev_summaries = {
#             'xent': self.los['xen'],
#             'xent_smooth': self.los['xens'],
#             'acc': self.acc['accgls'],
#         }
#         return dev_summaries
        
#     def get_scoring(self):
#         """ Get scoring function (used to rank models, lower is better). """
#         return self.los['xen']
    
#     def get_inputs_handler(self):
#         """ Get handler function for input batch. """
#         return lambda x,y: (x, y[:-1])
    
#     def get_output_handler(self):
#         """ Get handler function for output batch. """
#         return lambda x,y: y.view(-1)
    
#     def get_prediction_handler(self):
#         """ Get handler function for output batch. """
#         return lambda p: p.view(-1, p.size(-1))

    
# class DefmodBaseXens(DefmodBaseXen):
#     def __init__(self):
#         super().__init__()
    
#     def get_criterion(self):
#         """ Get criterion - for loss. """
#         return self.los['xens']
    
#     def get_scoring(self):
#         """ Get scoring function (used to rank models, lower is better). """
#         return self.los['xens']


    
# SETTINGS = {
#     # Embed2embed settings
#     'embed2embed-mlp-mse': Embed2embedMlpMse, # MSE
#     'embed2embed-mlp-mse-cos5': Embed2embedMlpMseCos5, # MSE + 0.5*COS
#     # Revdict settings
#     'revdict-base-mse': RevdictBaseMse, # MSE
#     # Defmod settings
#     'defmod-base-xen': DefmodBaseXen, # Cros-entropy
#     'defmod-base-xens': DefmodBaseXens, # Cros-entropy smooth labels
# }