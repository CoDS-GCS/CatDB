import shutil

from metagpt.ext.sela.evaluation.evaluation import node_evaluate_score_sela
from metagpt.ext.sela.evaluation.visualize_mcts import get_tree_text
from metagpt.ext.sela.runner.runner import Runner
from metagpt.ext.sela.search.search_algorithm import MCTS, Greedy, Random
import time
from .LogResults import LogResults


class MCTSRunner(Runner):

    def __init__(self, args, tree_mode=None, **kwargs):
        self.start_task_id = args.start_task_id
        self.result_path = kwargs.get("result_output_path", None)
        self.dataset_name = kwargs.get("task", None)
        self.task_type = kwargs.get("task_type", None)
        self.number_iteration = kwargs.get("rollouts", None)
        self.llm_model = kwargs.get("llm_model", None)
        

        super().__init__(args, **kwargs)
        self.tree_mode = tree_mode

    async def run_experiment(self):
        time_start = time.time()
        
        use_fixed_insights = self.args.use_fixed_insights
        depth = self.args.max_depth
        if self.tree_mode == "greedy":
            mcts = Greedy(root_node=None, max_depth=depth, use_fixed_insights=use_fixed_insights)
        elif self.tree_mode == "random":
            mcts = Random(root_node=None, max_depth=depth, use_fixed_insights=use_fixed_insights)
        else:
            mcts = MCTS(root_node=None, max_depth=depth, use_fixed_insights=use_fixed_insights)
        best_nodes = await mcts.search(state=self.state, args=self.args)
        best_node = best_nodes["global_best"]
        dev_best_node = best_nodes["dev_best"]
        score_dict = best_nodes["scores"]
        log_results = LogResults(dataset_name=self.dataset_name,
                                 task_type=self.task_type,                                 
                                 classifier="Auto",
                                 status="True",                                 
                                 number_iteration=self.number_iteration,
                                 has_description="No",
                                 llm_model=self.llm_model,
                                 config="SELA",
                                 sub_task="",
                                 number_iteration_error=0,
                                 time_catalog_load=0)
        
        node_evaluate_score_sela(node=best_node, task_type=self.task_type, log_results=log_results)
        # additional_scores = {"grader": node_evaluate_score_sela(node=dev_best_node, task_type=self.task_type, log_results=log_results)}

        # text, num_generated_codes = get_tree_text(mcts.root_node)
        # text += f"Generated {num_generated_codes} unique codes.\n"
        # text += f"Best node: {best_node.id}, score: {best_node.raw_reward}\n"
        # text += f"Dev best node: {dev_best_node.id}, score: {dev_best_node.raw_reward}\n"
        # text += f"Grader score: {additional_scores['grader']}\n"
        # print(text)
        # results = [
        #     {
        #         "best_node": best_node.id,
        #         "best_node_score": best_node.raw_reward,
        #         "dev_best_node": dev_best_node.id,
        #         "dev_best_node_score": dev_best_node.raw_reward,
        #         "num_generated_codes": num_generated_codes,
        #         "user_requirement": best_node.state["requirement"],
        #         "tree_text": text,
        #         "args": vars(self.args),
        #         "scores": score_dict,
        #         "additional_scores": additional_scores,
        #     }
        # ]
        # self.save_result(results)
        # self.copy_notebook(best_node, "best")
        # self.copy_notebook(dev_best_node, "dev_best")
        # self.save_tree(text)

        time_end = time.time()
        log_results.time_total = time_start - time_end
        log_results.time_execution=log_results.time_total
        log_results.time_pipeline_generate = log_results.time_total
        log_results.save_results(result_output_path= self.result_path)
        
        # prompt_token_count=caafe_clf.prompt_number_of_tokens,
        # all_token_count=performance['number_of_tokens']
    def copy_notebook(self, node, name):
        node_dir = node.get_node_dir()
        node_nb_dir = f"{node_dir}/Node-{node.id}.ipynb"
        save_name = self.get_save_name()
        copy_nb_dir = f"{self.result_path}/{save_name}_{name}.ipynb"
        shutil.copy(node_nb_dir, copy_nb_dir)

    def save_tree(self, tree_text):
        save_name = self.get_save_name()
        fpath = f"{self.result_path}/{save_name}_tree.txt"
        with open(fpath, "w") as f:
            f.write(tree_text)
