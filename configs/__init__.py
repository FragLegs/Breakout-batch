# import importlib
import os
# import pkgutil

# # import all modules in the "configs" package
# # and add references to them in a dict, keyed on their names
# CONFIG_MAP = {
#     mod_name: importlib.import_module('{}.{}'.format(__name__, mod_name))
#     for _, mod_name, _ in pkgutil.iter_modules(
#         [os.path.dirname(os.path.abspath(__file__))]
#     )
# }


class Config(object):
    def __init__(self,
                 env_name,
                 explore_name,
                 batch,
                 run_id,
                 alt_name,
                 **kw):
        # env config
        self.render_train     = False
        self.render_test      = False
        self.env_name         = env_name
        self.overwrite_render = True
        self.record           = True
        self.high             = 255.

        # output config
        self.output_path  = (
            "{working_dir}/results/{env}/{explore}{batch}/{run_id}/".format(
                working_dir=os.environ.get('DOMINO_WORKING_DIR', 'domino'),
                env=self.env_name,
                explore=explore_name if alt_name is None else alt_name,
                batch='Batch' if batch else '',
                run_id=run_id,
            )
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path + "monitor/"

        if batch:
            non_batch_config = Config(
                env_name=env_name,
                explore_name=explore_name,
                batch=False,
                run_id=run_id,
                alt_name=alt_name
            )
            self.buffer_path = non_batch_config.buffer_path
        else:
            self.buffer_path  = self.output_path + "buffer.npz"

        # model and training config
        model_name = 'DQN'
        if not batch and 'Count' in explore_name:
            model_name = 'CountingDQN'

        self.model             = model_name
        self.explore           = explore_name
        self.batch             = batch
        self.num_episodes_test = 50
        self.grad_clip         = True
        self.clip_val          = 10
        self.saving_freq       = 250000
        self.log_freq          = 50
        self.eval_freq         = 50000 if batch else 250000
        self.record_freq       = 250000
        self.soft_epsilon      = 0.05

        # nature paper hyper params
        self.nsteps_train       = 2500000
        self.batch_size         = 32
        self.buffer_size        = self.nsteps_train
        self.target_update_freq = 10000
        self.gamma              = 0.99
        self.learning_freq      = 4
        self.state_history      = 4
        self.skip_frame         = 4
        self.lr_begin           = 0.00025
        self.lr_end             = 0.00005
        self.lr_nsteps          = self.nsteps_train / 2
        self.eps_begin          = 1
        self.eps_end            = (
            0.5 if (alt_name is not None and 'me' in alt_name) else 0.1
        )
        self.eps_nsteps         = 1000000
        self.learning_start     = 50000

        # For counting
        self.sim_hash_k         = 256
        self.beta               = 1.0
