import requests

host_link = "http://localhost:8989/"


def test_using_post():
    json_data = [[2679000.0, 1248000.0], [2679000.0, 1250000.0], [2681000.0, 1250000.0], [2681000.0, 1248000.0]]
    r = requests.post(host_link + "construct_graph?project_name=test", json=json_data)
    # print status - should be 200
    print(r.status_code)
    # print output data
    print(r.json())


test_using_post()
