---
date: 2026-06-12
categories:
  - releases
  - leaderboard
authors:
  - samoed
  - kenneth
  - isaac
description: >
  We released a new version of the MTEB leaderboard that is miles faster, while improving filtering, model comparison, and transparency allows you to dig deep into which model is right for you.
---
*TL:DR*: We released a new version of the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) that is miles faster, while improving filtering, model comparison, and transparency allows you to dig deep into which model is right for you.

<!-- more -->

# MTEB Leaderboard: From a slow demo to feature-rich leaderboard

*TL:DR*: We released a new version of the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) that is miles faster, while improving filtering, model comparison, and transparency allows you to dig deep into which model is right for you.

![hero-1-1](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/j5XqMjx34l3plTPqtY5Qz.png)

Since its initial release, MTEB has had multiple benchmarks, starting from a simple table view with minimal ability to filter to a granular leaderboard. While initially reasonably fast, with the increase in both the number of models and benchmarks mteb covers then benchmark has become unreliable — both in terms of speed \[[1](https://huggingface.co/spaces/mteb/leaderboard/discussions/185), [2](https://huggingface.co/spaces/mteb/leaderboard/discussions/182), [3](https://github.com/embeddings-benchmark/mteb/issues/4411)\] and uptime \[[4](https://github.com/embeddings-benchmark/mteb/issues/4709), [5](https://github.com/embeddings-benchmark/mteb/issues/4273#issuecomment-4106896053)\]. This has been frustrating for both us as developers and the users. With this release, we are very happy to share a new leaderboard built on a more reliable and scalable framework using FastAPI and Svelte, enabling us to greatly improve the user experience — both in terms of current speed and features- and it will also enable us to deliver leaderboard improvements continually in the future.

In this blog, we will go through some of the highlights, including design decisions focusing on speed, transparency, and improved tooling to help you select more models.

![leaderboard-versions-comparison](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/8sNVgjhcvhVoMNIzNFgXa.png)


## It is fast!

To address the elephant in the room, [the new leaderboard](https://huggingface.co/spaces/mteb/leaderboard) is miles faster than the previous benchmarks — we could go into details on this, but while it is important, you can find many blog posts on how to speed up a frontend, and if you want more proof, you can just go try it out — you can even explore the leaderboard and benchmarks from your phone!


## Encouraging Exploration and Customization

When selecting a benchmark, it is worth knowing that it rarely measures what you care about. A pre-defined leaderboard might only contain 50% of the tasks that you care about, and while model performance (roughly) tends to correlate across tasks, you can often do a lot better by customizing the leaderboard specifically to your use case. You can see more information about models by hovering over their names and pin models for easy comparison.

![task-tip](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/VsupwayvHgjeJgCruZ_Sx.png)

We have now made this much easier, with benchmark filters allowing you to filter on specific domains task and modalities and with the ability to learn more about each specific task to tailor it to your needs. Filters allow filtering on domains, language, modality, and even individual tasks.

![benchmark-filters](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/e6KiWQbniNuuwjirN0eUh.png)

You can also explore [models](https://mteb-leaderboard.hf.space/models), [tasks](https://mteb-leaderboard.hf.space/tasks) and their results.

![grid-overview](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/6AUVXPQuG73gAl91ts2GL.png)

But what if that is not enough? That brings us to the next section.


## Transparency

It should be easy to dig into the datasets used to evaluate our models, but many benchmarks make this hard, perhaps because they often contain errors [[6](https://arxiv.org/abs/2103.14749), [7](https://arxiv.org/abs/2511.16842), [8](https://aclanthology.org/2025.naacl-long.262/)]. However, if we want people to trust our tools, transparency is the key; the popularity of [bullshitbench](https://petergpt.github.io/bullshit-benchmark/viewer/index.v2.html) is an excellent example of this.

In [the new leaderboard](https://huggingface.co/spaces/mteb/leaderboard), we enable this by allowing you to inspect a task and by integrating a viewer for the huggingface datasets, along with the results and task metadata. 

![figure-trained-on](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/CSatoGNoh2g9U_-BtYA9Q.png)

![figure-task-detail](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/Zqa-h7Gy58hVbD2odCv1U.png)

As you can see in the table in the figure, we also indicate whether a model has been trained on the task's training set or is seeing it for the first time (zero-shot). You will see these annotations showing up throughout the benchmarks to clearly indicate potential issues.

## Improving to the frontier not just the top

A common problem (and feature) of benchmarks is that they rank models by performance, thereby encouraging development toward better performance while ignoring other factors such as size, memory usage, and runtime. We seek to encourage broader improvements across the frontier rather than just the top models, both by ensuring that quick views of the front page display the top models for their size bracket and, of course, by providing performance-by-runtime analytics. 

![figure-perf-size](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/QNjqG0NauRwn2pNG6iAMh.png)
![figure-primary-tile](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/DcpOdv5csrC9vlZ7cICAB.png)


## Compare Models

The ranking can help you get a quick overview of the models, but you will soon find the need to compare two specific models for your use-case. This is now easier than ever: simply pin the models you want to compare, and they will be reordered and highlighted for easy comparison.


If you want to dig even deeper into your comparison, simply press the button stating “compare {n} pinned” and you will be shown a head-to-head comparison of the models

![figure-compare copy](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/kdae7N-ICqr9kauDcSvZv.png)

![figure-compare](https://cdn-uploads.huggingface.co/production/uploads/61af4544d691b3aadd1f62b6/LFYJ0uYrrjQvtd2V_I7aG.png)

## API

If you need to fetch scores for the leaderboard locally, you can download a CSV or try our API: https://mteb-leaderboard-backend.hf.space/docs

## Thought of a good feature to add?

If you used the leaderboard or read the blog and came up with a good feature to add, feel free to suggest it as an enhancement [issue](https://github.com/embeddings-benchmark/mteb/issues/new?template=enhancement.yaml). Of course, if you find bugs, we are more than happy to hear about them as well. 

A big thanks to those who have already provided feedback and improvements along the way.
