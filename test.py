import requests


url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

#url  = "https://i351f1yczb.execute-api.us-west-2.amazonaws.com/test/predict"

#data = {"url": "https://user-images.githubusercontent.com/46135649/207101011-77379ccc-6684-4b74-93e4-852367e28920.png"}

data = {"url": "https://user-images.githubusercontent.com/46135649/208242713-15db7a6c-7b54-4c8f-9945-ca05edce705e.png"}

result = requests.post(url, json=data).json()

print(result)
