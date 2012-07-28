class dummy_directview(object):
    map_sync = map
    __len__ = lambda self: 1
    purge_results = lambda x,y: None
dv = dummy_directview()

def go_parallel():
    global dv, c
    from IPython.parallel import Client
    c = Client()
    dv = c[:]
