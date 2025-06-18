import numpy as np
class CrossEntropyLoss:
    def forward(self, predictions, labels):
        # we need to avoid  log(0) so we just put low values
        predictions = np.clip(predictions, 1e-12, 1. - 1e-12)


        if labels.ndim == 2:
            # if the predictions are hot encode, well we just multiply because all the other will be 0 so they wont be contributing to the summing, we sum by row
            loss = -np.sum(labels * np.log(predictions), axis=1)
        else:
            # If labels are integers, we then go thorugh every sample and find the value of the class and our probaibility
            loss = -np.log(predictions[np.arange(len(labels)), labels])

        return np.mean(loss)