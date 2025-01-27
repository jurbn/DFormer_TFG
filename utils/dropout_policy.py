
class PolicyDR():
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def get_drop_rate(self, epoch):
        # simply remove 10% every 100 epoch (min is 0)
        return max(self.dropout_rate - (epoch / 1000.0), 0.0)