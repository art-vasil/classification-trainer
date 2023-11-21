from src.model.estimator import ClassifierTrainer


if __name__ == '__main__':
    for i in range(2):
        ClassifierTrainer(para_idx=i).train()
