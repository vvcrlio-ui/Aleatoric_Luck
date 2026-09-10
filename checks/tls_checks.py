import shutil
import ssl
import subprocess
import threading
import urllib.error
import pytest
from aleatoric_nk_grid.shared_queue import Dispatcher, ModelTask
from aleatoric_nk_grid.queue_service import Client, make_server


def test_real_tls_trust_and_hostname_verification(tmp_path):
    openssl=shutil.which('openssl')
    if not openssl:pytest.skip('openssl executable required for temporary test certificate')
    key=tmp_path/'key.pem';cert=tmp_path/'cert.pem'
    subprocess.run([openssl,'req','-x509','-newkey','rsa:2048','-nodes','-days','1',
        '-keyout',str(key),'-out',str(cert),'-subj','/CN=localhost',
        '-addext','subjectAltName=DNS:localhost'],check=True,capture_output=True)
    ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(cert,key)
    Dispatcher.create(tmp_path/'queue',[(ModelTask(1,0,10,1,'ols'),1)],identity={'test':'TLS'})
    with Dispatcher(tmp_path/'queue',scratch=tmp_path/'scratch') as q:
        server=make_server(q,token='t'*32,tls_context=ctx)
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            client=Client(f'https://localhost:{server.server_port}','t'*32,q.queue_id,ca_file=str(cert))
            assert client.call('claim',worker='w')['state']=='task'
            untrusted=Client(f'https://localhost:{server.server_port}','t'*32,q.queue_id)
            with pytest.raises(urllib.error.URLError):untrusted.call('stats')
            wrong_host=Client(f'https://127.0.0.1:{server.server_port}','t'*32,q.queue_id,ca_file=str(cert))
            with pytest.raises(urllib.error.URLError):wrong_host.call('stats')
        finally:
            server.shutdown();server.server_close();thread.join()
