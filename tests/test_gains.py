from rpva.audit import compute_losses, compute_paired_gains


def test_paired_gain():
    rows=[{'event_id':1,'agent_id':'a','role':'x','information_state':'S0','y_true':2,'y_pred':0},{'event_id':1,'agent_id':'a','role':'x','information_state':'S1','y_true':2,'y_pred':1}]
    gains=compute_paired_gains(compute_losses(rows), ['S0','S1'])
    assert gains[0]['gain'] == 3.0
