# import logging

# import mteb

# logging.basicConfig(level=logging.INFO, format="%(message)s")


# def main():
#     model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
#     task = mteb.get_task("SciFact")
#     results = mteb.evaluate(
#         model, task, overwrite_strategy="always", encode_kwargs={"batch_size": 32}
#     )
#     task_result = results.task_results[0]
#     print("evaluation_time:", task_result.evaluation_time)
#     print("\n--- Plotting Evaluation Phases ---")
#     task_result.plot_evaluation_phases()


# if __name__ == "__main__":
#     main()


# import mteb

# model = mteb.get_model("mteb/hybrid-baseline_encoder-e5-small")
# task = mteb.get_task("SciFact")
# results = mteb.evaluate(model, task)
# task_result = results.task_results[0]
# print(task_result)
