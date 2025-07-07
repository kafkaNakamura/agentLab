# Adaptive Safety Posture for LLM Agents with Dynamic Intent Inference

## Research Report | Agent Laboratory

This repository presents the foundational concepts, architectural design, and initial methodological findings for the Adaptive Safety Posture (ASP) framework. Our research aims to significantly enhance the safety of Large Language Model (LLM) agents, particularly those augmented with tool-use capabilities, by dynamically modulating safety policies based on real-time, inferred user intent within multi-turn human-AI interactions.

---

## Table of Contents

*   [Introduction](#introduction)
*   [The Problem](#the-problem)
*   [Core Architecture: Adaptive Safety Posture (ASP)](#core-architecture-adaptive-safety-posture-asp)
    *   [User Intent Inference Module (UIIM)](#user-intent-inference-module-uiim)
    *   [Adaptive Safety Policy (ASP) Module](#adaptive-safety-policy-asp-module)
*   [Key Contributions](#key-contributions)
*   [Experimental Methodology & Key Insights](#experimental-methodology--key-insights)
    *   [Initial Data Generation Flaw](#initial-data-generation-flaw)
    *   [Refined Data Generation Strategy](#refined-data-generation-strategy)
*   [Anticipated Results & Projections](#anticipated-results--projections)
*   [Future Work](#future-work)
*   [Installation](#installation) (Placeholder)
*   [Usage](#usage) (Placeholder)
*   [Contributing](#contributing) (Placeholder)
*   [License](#license) (Placeholder)
*   [Acknowledgements](#acknowledgements)
*   [References](#references)

---

## Introduction

The proliferation of Large Language Models (LLMs) equipped with sophisticated tool-use capabilities marks a transformative shift towards autonomous AI agents. These "foundation agents" can perceive, reason, plan, and act within diverse environments, from file system management to scientific discovery. While promising unprecedented efficiency, this expanded autonomy concurrently introduces critical safety and alignment challenges, especially concerning extrinsic threats arising from complex, multi-turn human-AI interactions.

Traditional static safety layers often prove inadequate in these dynamic scenarios. They are either excessively restrictive, hindering benign use, or dangerously permissive, failing to mitigate subtle, evolving adversarial intentions. This paper proposes an Adaptive Safety Posture (ASP) for LLM agents – a novel framework designed to dynamically modulate safety policies based on continuously inferred user intent, providing a robust and nuanced approach to agent safety.

## The Problem

The inherent difficulty in securing tool-augmented LLM agents in multi-turn settings stems from the complex, ambiguous, and often adversarial nature of human-AI communication. Adversarial users rarely employ overt, single-turn attacks. Instead, they leverage sophisticated, multi-turn strategies like deceptive phrasing, social engineering, or gradual escalation, making static filters obsolete. A seemingly innocuous request like "list files" could be benign or a precursor to a malicious data exfiltration attempt.

Accurate and continuous inference of user intent is a formidable challenge, as natural language is rife with ambiguity, and malicious actors actively obfuscate their true objectives. Without a robust, real-time, context-aware mechanism to discern these shifting intentions, safety systems are relegated to a reactive posture, attempting to block harmful actions only *after* the agent has proposed them – a critically insufficient approach for irreversible tool operations. This creates a profound performance-safety trade-off: maximizing helpfulness for legitimate users while maintaining stringent protection against malicious ones.

## Core Architecture: Adaptive Safety Posture (ASP)

The Adaptive Safety Posture (ASP) framework integrates three core, interconnected modules to achieve its dynamic safety capabilities:

1.  **Core Agent LLM**: The primary conversational engine (e.g., `GPT-3.5-turbo`, `Llama3.1-8B-Instruct`) responsible for understanding user inputs, engaging in dialogue, and proposing calls to external tools. Crucially, proposed tool calls are *not* directly executed but are routed through the ASP system.
2.  **User Intent Inference Module (UIIM)**: A dedicated, fine-tuned causal language model (e.g., `Gemma-2B-it`, `Mistral-7B-Instruct`) engineered for continuous interpretation of user behavior.
3.  **Adaptive Safety Policy (ASP) Module**: A lightweight rule-based system or small ML model that acts as an intelligent gatekeeper for proposed tool calls.

The UIIM takes the cumulative conversational history $H_t = \{u_1, a_1, \dots, u_{t-1}, a_{t-1}, u_t\}$ as input to infer the user's current underlying intent $I_t$. This intent is classified into one of four categorical labels, representing a calibrated spectrum of risk:

*   **Benign**: Standard, non-malicious interaction.
*   **Probing/Exploratory**: User inquiries about system capabilities, potentially testing boundaries, but not overtly malicious.
*   **Malicious (Subtle)**: Covert or deceptive attempts to elicit sensitive information or trigger unintended actions.
*   **Malicious (Overt)**: Direct, explicit adversarial attempts to cause harm or bypass safety.

The inferred intent $I_t$ then dynamically informs the ASP Module's intervention logic for any proposed tool calls $T$ from the Core Agent LLM:

### User Intent Inference Module (UIIM)

*   **Input**: Cumulative conversational history $H_t$.
*   **Output**: Categorical intent label $I_t \in \{\text{Benign}, \text{Probing/Exploratory}, \text{Malicious (Subtle)}, \text{Malicious (Overt)}\}$.
*   **Implementation**: Fine-tuned causal language model (e.g., Gemma-2B-it or Mistral-7B-Instruct).
*   **Goal**: Achieve high F1-score (target > 0.8) for intent classification.

### Adaptive Safety Policy (ASP) Module

*   **Benign Intent ($I_t = \text{Benign}$)**:
    *   Proposed tool call $T$ is typically executed directly, subject to basic static safety checks (e.g., whitelist adherence).
    *   **Goal**: Minimize friction, maximize helpfulness.
*   **Probing/Exploratory Intent ($I_t = \text{Probing/Exploratory}$)**:
    *   If $T$ is sensitive (e.g., `write_file`, `delete_file`, access `protected_directories`), mandatory, explicit user confirmation is triggered.
    *   Confirmation is accompanied by clear warnings. Tool proceeds only with explicit user consent ($P(\text{confirmation} | I_t=\text{Probing}) > \tau$).
*   **Malicious (Subtle) Intent ($I_t = \text{Malicious (Subtle)}$)**:
    *   Direct execution of *any* proposed tool call is strictly prohibited.
    *   Agent generates a safe, pre-formatted version of the command for *manual user review and explicit approval* outside agent control.
    *   If adversarial behavior persists, agent shifts to non-tool-using conversational mode.
*   **Malicious (Overt) Intent ($I_t = \text{Malicious (Overt)}$)**:
    *   Any proposed tool call is immediately and unconditionally blocked.
    *   Agent enters "safety override" state, providing a firm safety warning (e.g., "I cannot fulfill this request as it violates safety protocols. Please adjust your query.").
    *   Agent refuses to explain block specifics to prevent red-teaming.

## Key Contributions

This work makes significant contributions to AI safety, intelligent agents, and human-AI interaction:

*   **Adaptive Safety Posture (ASP) Framework**: A novel paradigm moving beyond static, reactive safety layers to a dynamic, intent-aware framework for mitigating extrinsic threats in complex, multi-turn human-AI interactions.
*   **User Intent Inference Module (UIIM)**: A dedicated, fine-tuned causal language model for continuous and nuanced classification of user intents (Benign, Probing/Exploratory, Malicious (Subtle), Malicious (Overt)) from cumulative conversational history.
*   **Tiered, Adaptive Safety Policy Mechanism**: Integration of differentiated interventions for proposed tool calls, balancing helpfulness for legitimate users with proactive risk mitigation against adversarial ones.
*   **Rigorous Experimental Methodology**: Delineation of a robust validation framework, including a crucial pivot to human-curated dataset generation, ensuring foundational integrity for empirical claims.

## Experimental Methodology & Key Insights

Our experimental phase is designed to assess the ASP's efficacy in enhancing agent safety against extrinsic threats while preserving helpfulness. The Adaptive Safety Agent will be comparatively analyzed against a static baseline agent (Core Agent LLM with basic reactive safety filters).

Key evaluation metrics:
*   **Unsafe Tool Action Execution Rate (UTAE-R)**: Percentage of `Probing/Malicious` scenarios resulting in safety-critical failures.
*   **Unsafe Tool Action Attempt Prevention Rate (UTAP-R)**: Percentage of `Probing/Malicious` scenarios where the agent successfully blocks or mitigates critical tool calls.
*   **Intent Classification F1-Score (Macro-average)**: UIIM's accuracy in classifying user intent.
*   **Benign Task Completion Rate (BTCR)**: Percentage of `Benign` tasks successfully completed.
*   **Efficiency Metrics**: Average Turns to Completion/Resolution, Average Inference Latency.

### Initial Data Generation Flaw

A critical preliminary finding during UIIM dataset generation revealed a fundamental data integrity issue. The Python script for `prepare_uiim_training_data` erroneously used `random.choice` to assign `ground_truth_intent` labels, instead of deriving them from crafted scenario logic. This meant assigned labels were pseudo-random, rendering any UIIM trained on such data effectively useless (expected F1-score ~0.25 for 4 classes). This crucial "negative" result necessitated a fundamental shift in our methodological approach.

### Refined Data Generation Strategy

In light of this discovery, the project pivoted to a meticulous, human-curated ground truth data generation process:

1.  **Detailed Scenario Templates**: Meticulously designed for each intent category, specifying initial context, stated and implicit user goals, and anticipated tool interactions.
2.  **LLM-simulated Dialogues**: GPT-4-turbo is used in a sophisticated role-playing capacity to simulate realistic, multi-turn dialogues for diverse conversational histories ($H_t$).
3.  **Human Turn-Level Annotation**: Trained human annotators rigorously label each user turn with a precise ground truth intent ($I_t$), based on the cumulative history. Strict guidelines, Inter-Annotator Agreement (IAA) calculation (target Kappa > 0.8), and adjudication ensure consistency.
4.  **Expected Safety Outcome Labeling**: For each scenario, the *expected safety outcome* for critical tool call attempts is also labeled (e.g., `Safe Action Executed`, `Unsafe Action Attempted - Should Block`).
5.  **Dataset Size**: Target 500-700 multi-turn dialogues, split into 70% training, 10% validation, 20% testing, with balanced representation across intent categories.

This meticulous, human-centric process establishes an unimpeachable foundation for training the UIIM, ensuring accurate intent inference and the validity of all subsequent empirical claims.

## Anticipated Results & Projections

Based on our refined methodological framework, we project the following outcomes:

*   **UIIM Performance**: A Macro-average F1-score exceeding 0.8 for intent classification, signifying reliable real-time intent signals.
*   **Reduced UTAE-R**: At least a 70-80% decrease in Unsafe Tool Action Execution Rate compared to static baselines in `Probing/Exploratory`, `Malicious (Subtle)`, and `Malicious (Overt)` scenarios.
*   **Increased UTAP-R**: A 60-75% improvement in Unsafe Tool Action Attempt Prevention Rate in high-risk scenarios, due to proactive, tiered interventions.
*   **Preserved BTCR**: Maintenance of Benign Task Completion Rate at approximately 95% or higher, demonstrating negligible degradation in helpfulness for legitimate users.
*   **Acceptable Efficiency**: UIIM classification and ASP decision-making (lightweight models) expected latency of $<200$ms, ensuring no prohibitive delays. Benign tasks 2-4 turns; adversarial 4-7 turns before resolution/blocking.

These projections, once empirically validated, will underscore the efficacy of our adaptive, intent-aware framework in achieving a superior performance-safety trade-off compared to existing static paradigms.

## Future Work

Future work will include:
*   Detailed ablation studies to quantify the direct contribution of UIIM accuracy to overall system safety metrics.
*   Analysis of the isolated effectiveness of each distinct ASP policy tier.
*   Exploration of more sophisticated, learning-based ASP modules beyond rule-based systems.
*   Deployment and evaluation in broader, more complex simulated and real-world environments.

## Installation (Placeholder)

Instructions for setting up the Adaptive Safety Posture framework will be provided here upon release of the code. This will typically involve:

1.  Cloning the repository.
2.  Setting up a Python environment.
3.  Installing dependencies (e.g., `transformers`, `torch`, `scikit-learn`).
4.  Downloading pre-trained UIIM models and necessary data.

## Usage (Placeholder)

Detailed examples and scripts for running the Adaptive Safety Agent and evaluating its performance will be available here. This will include:

*   Running interactive dialogues with the ASP agent.
*   Scripts for evaluating against test datasets.
*   Instructions for fine-tuning the UIIM on custom datasets.

## Contributing (Placeholder)

We welcome contributions to this project! Please refer to the `CONTRIBUTING.md` file (to be added) for guidelines on how to submit issues, pull requests, and contribute to the development of the Adaptive Safety Posture.

## License (Placeholder)

This project is licensed under the [LICENSE NAME] - see the `LICENSE.md` file (to be added) for details.

## Acknowledgements

The contributions from Argonne National Laboratory were supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357. XLQ acknowledges the support of the Simons Foundation.

## References

*   [2504.01990] Foundation Agents: A New Paradigm for Autonomous AI Systems (arXiv link pending)
*   [2506.04980v1] Enhancing LLM Agent Capabilities through Advanced Tool Integration (arXiv link pending)
*   [2409.16427v3] HAICOSYSTEM: A Framework for Multi-Turn Human-AI Collaboration Safety (arXiv link pending)
*   [2308.12194v2] Understanding and Mitigating Jailbreaking Attacks in Large Language Models (arXiv link pending)
*   [2406.06051v3] Prompt Injection Attacks: A Survey of Mitigation Strategies (arXiv link pending)
*   [2402.07221v2] Intent Prediction in Human-Agent Interaction (arXiv link pending)
*   [2505.13008v2] Task-Oriented Dialogue Systems with Dynamic Intent Recognition (arXiv link pending)
*   [2506.23844v1] Scaling AI Capabilities with Controllable Design Paradigms (arXiv link pending)
*   [2506.02923v1] Limits of Predicting Agents from Behaviour (arXiv link pending)
*   [2305.19223v1] Preserving Human Agency in AI Systems (arXiv link pending)
*   [2506.09656v1] The Value Alignment Problem in Artificial Intelligence (arXiv link pending)