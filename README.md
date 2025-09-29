
<div align="center">

<h1>MASLegalBench: Benchmarking Multi-Agent Systems in Deductive Legal Reasoning</h1>

<p>
  <a href="https://egbertjing.github.io/">Huihao Jing</a><sup>😎 1</sup>, 
  <a href="https://whuak.github.io/">Wenbin Hu</a><sup>😎 1</sup>, 
  Hongyu Luo<sup>1</sup>, 
  Jianhui Yang<sup>2</sup>, 
  <a href="https://alexfan.cn/">Wei Fan<sup>1</sup>, 
  <a href="https://hlibt.student.ust.hk/">Haoran Li</a><sup>🤗 1</sup>, 
  <a href="https://www.cse.ust.hk/~yqsong/">Yangqiu Song</a><sup>1</sup>
</p>

<p>
<sup>😎</sup>Equal contribution.
<sup>🤗</sup>Corresponding author.  
</p>

<p>
<sup>1</sup>Hong Kong University of Science and Technology  
<br>
<sup>2</sup>Tsinghua University
</p>

</div>


<p align="center">
  <!-- <a href='https://github.com/HKUST-KnowComp/MASLegalBench'>
  <img src='https://img.shields.io/badge/Arxiv-2405.20340-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>  -->
  <a href='https://github.com/HKUST-KnowComp/MASLegalBench'>
  <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a> 
  <a href='https://github.com/HKUST-KnowComp/MASLegalBench'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
  <a href='LICENSE'>
  <img src='https://img.shields.io/badge/License-IDEA-blue.svg'>
  </a> 
  <a href="" target='_blank'>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=HKUST-KnowComp.MASLegalBench&left_color=gray&right_color=%2342b983">
  </a> 
</p>


# 🤩 Abstract

Multi-agent systems (MAS), leveraging the remarkable capabilities of Large Language Models (LLMs), show great potential in addressing complex tasks. In this context, integrating MAS with legal tasks is a crucial step. While previous studies have developed legal benchmarks for LLM agents, none are specifically designed to consider the unique advantages of MAS, such as task decomposition, agent specialization, and flexible training. In fact, the lack of evaluation methods limits the potential of MAS in the legal domain. To address this gap, we propose MASLegalBench, a legal benchmark tailored for MAS and designed with a deductive reasoning approach. Our benchmark uses GDPR as the application scenario, encompassing extensive background knowledge and covering complex reasoning processes that effectively reflect the intricacies of real-world legal situations. Furthermore, we manually design various role-based MAS and conduct extensive experiments using different state-of-the-art LLMs. Our results highlight the strengths, limitations, and potential areas for improvement of existing models and MAS architectures.

# Miscellaneous
Please send any questions about the code and/or the method to hjingaa@connect.ust.hk
<div align="center">


# 😎 Quick Start: Try Our MCIP Guardian Model
```
bash scripts/bm25/eval_qwen25.sh
bash scripts/bm25/eval_qwen3_8b.sh
...
```

<pre>
███╗   ███╗ █████╗ ███████╗██╗      ███████╗ ██████╗  █████╗ ██╗     ██████╗  ███████╗███╗   ██╗ ██████╗██╗   ██╗
████╗ ████║██╔══██╗██╔════╝██║      ██╔════╝██╔════╝ ██╔══██╗██║     ██╔══██╗ ██╔════╝████╗  ██║██╔════╝██║   ██║
██╔████╔██║███████║███████╗██║      █████╗  ██║  ███╗███████║██║     ██████╔╝ █████╗  ██╔██╗ ██║██║     ████████║
██║╚██╔╝██║██╔══██║╚════██║██║      ██╔══╝  ██║   ██║██╔══██║██║     ██╔══██╗ ██╔══╝  ██║╚██╗██║██║     ██╔═══██║
██║ ╚═╝ ██║██║  ██║███████║███████╗ ███████╗╚██████╔╝██║  ██║███████╗██████╔╝ ███████╗██║ ╚████║╚██████╗██║   ██║
╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═════╝  ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝   ╚═╝
<pre>
</div>
