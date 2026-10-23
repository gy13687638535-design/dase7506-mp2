"""Course interface for the frozen round-2 selector."""
def predict(bank, arrays, checkpoint):
    from round2_model import predict as select
    return select(bank, arrays, checkpoint)
