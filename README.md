# NOBERT

NOBERT is a benchmarking framework that evaluates the performance of pretrained transformer models against conventional feature-based machine learning methods for regulatory genomics tasks. The framework enables fair comparisons by using identical input features across models.

## Workflow

1. **Pretraining:** Pretrain a BERT-based transformer on cistrome profiles.
2. **Fine-tuning:** Fine-tune the pretrained model on downstream regulatory prediction tasks.
3. **Baseline Models:** Train conventional feature-based machine learning models on the same downstream tasks using identical input features.
4. **Performance Evaluation:** Compare predictive performance across all methods.
5. **Feature Importance Analysis:** Identify important regulatory features using model interpretation methods.
6. **Biological Interpretation:** Compare the regulatory features highlighted by different models and examine their biological relevance.
