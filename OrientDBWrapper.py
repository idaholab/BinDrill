import json 
import requests 
import logging

# Written orginally by Taylor McCampbell <taylor.mccampbell@inl.gov>, modified by Michael Cutshaw
class RestAPIWrapper:
   
    def __init__(self, host='localhost', port='2480', user='root', password='toor', db='test'):
        self.db = db
        self.host = host
        self.port = port 
        self.user = user 
        self.password = password

    def connectToDb(self):
        url = f'http://{self.host}:{self.port}/connect/{self.db}'
        try: 
            response = requests.get(url, auth=(self.user, self.password))
        except Exception as e:
            logging.error('Problem with http request')
            logging.exception(e)
        
        return response
        
    def query(self, queryString, limit=500):
        url = f'http://{self.host}:{self.port}/query/{self.db}/sql/{queryString}/{limit}'
        try: 
            response = requests.get(url, auth=(self.user, self.password))
        except Exception as e:
            logging.error('Problem with http request')
            logging.exception(e)
        
        return response.json()

    def listDatabases(self):
        url = f'http://{self.host}:{self.port}/listDatabases'

        try:
            response = requests.get(url, auth=(self.user, self.password))
            jsonResponse = response.json()
            return jsonResponse['databases']
        except Exception as e:
            logging.error('Problem with http request')
            logging.exception(e)

    def checkDbExists(self):
        return self.db in self.listDatabases()


if __name__ == "__main__":
    rapi = RestAPIWrapper(password="root", db="r3")
    print(rapi.query("select * from V"))