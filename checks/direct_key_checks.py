import pytest
from aleatoric_nk_grid.direct_key_resume import add_key, missing_tasks
from aleatoric_nk_grid.pending_resume import Design
from aleatoric_nk_grid.shared_queue import QueueError

class Cost:
    def estimate(self, *args): return 1.

def test_missing_is_exact_five_key_complement():
    spec={'resolved_k_grid':[2,9],'resolved_n_grid':[10,20],
          'resolved_repeat_plan':[[12345,0],[12345,1],[12346,0]],'models':['ols','ridge']}
    design=Design(spec)
    complete={0,7,8,15,23}
    all_tasks=[task for task,_ in missing_tasks(spec,design.bits,Cost())]
    for i in complete: add_key(design.bits,i,design.count)
    remaining=[task for task,_ in missing_tasks(spec,design.bits,Cost())]
    assert len(remaining)==19
    assert {t.id for t in remaining}=={t.id for i,t in enumerate(all_tasks) if i not in complete}
    assert [design.ordinal(t.__dict__) for t in all_tasks]==list(range(24))

def test_duplicate_and_out_of_design_are_rejected():
    bits=bytearray(2)
    add_key(bits,8,9)
    with pytest.raises(QueueError): add_key(bits,8,9)
    for key in [-1,9]:
        with pytest.raises(QueueError): add_key(bits,key,9)

def test_empty_and_full_completed_design():
    spec={'resolved_k_grid':[2],'resolved_n_grid':[10],
          'resolved_repeat_plan':[[12345,0]],'models':['ols','ridge']}
    d=Design(spec)
    assert len(list(missing_tasks(spec,d.bits,Cost())))==2
    for i in range(d.count): add_key(d.bits,i,d.count)
    assert list(missing_tasks(spec,d.bits,Cost()))==[]
