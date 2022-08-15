import requests

""" Modified class to make requests to EarthData (override de rebuild_auth)
"""
class SessionWithHeaderRedirection(requests.Session):

    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):
        super().__init__()

        self.auth = (username, password)

    def rebuild_auth(self, prepared_request, response):

        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:

            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and \
                redirect_parsed.hostname != self.AUTH_HOST and \
                original_parsed.hostname != self.AUTH_HOST:

                del headers['Authorization']
        return

session = SessionWithHeaderRedirection(username="", password="")

def run_api(url, idx):

    # save the file
    path = '../data/maiac/'
    # extract the filename from the url to be used when saving the file
    file_name = url[url.rfind('/')+1:]  # rfind() finds the last occurrence of the specified value.
    file_path = path + file_name

    try:
        # submit the request using the session
        response = session.get(url, stream=True)
        print(response.status_code)

        # raise an exception in case of http errors
        response.raise_for_status()  

        with open(file_path, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

        print(f"File {idx} saved!")

    except requests.exceptions.HTTPError as e:
        # handle any errors here
        print(e)