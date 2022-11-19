import logging.handlers
import logging
from __secrets import secrets
from typing import Any

class DiscordHandler(logging.handlers.HTTPHandler):
    
    def __init__(self, host, url, method="GET", secure=False, credentials=None,context=None):
        super().__init__(host, secrets[url], method, secure, credentials, context)
    
    def mapLogRecord(self, record: logging.LogRecord) -> 'dict[str, Any]':
        formatted = self.format(record)
        avatar_url = "https://uxwing.com/wp-content/themes/uxwing/download/17-internet-network-technology/artificial-intelligence-ai.png"
        content = f"{secrets['discord_ping_role']} {formatted}"
        return {"content": content, "avatar_url":avatar_url, "username":"SuperDan42" }
    
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a record.

        Send the record to the Web server as a percent-encoded dictionary
        """
        try:
            import http.client, json
            host = self.host
            if self.secure:
                h = http.client.HTTPSConnection(host, context=self.context)
            else:
                h = http.client.HTTPConnection(host)
            url = self.url
            data = json.dumps(self.mapLogRecord(record))
            if self.method == "GET":
                if (url.find('?') >= 0):
                    sep = '&'
                else:
                    sep = '?'
                url = url + "%c%s" % (sep, data)
            h.putrequest(self.method, url)
            # support multiple hosts on one IP address...
            # need to strip optional :port from host, if present
            i = host.find(":")
            if i >= 0:
                host = host[:i]
            # See issue #30904: putrequest call above already adds this header
            # on Python 3.x.
            # h.putheader("Host", host)
            if self.method == "POST":
                h.putheader("Content-type",
                            "application/json")
                h.putheader("Content-length", str(len(data)))
            if self.credentials:
                import base64
                s = ('%s:%s' % self.credentials).encode('utf-8')
                s = 'Basic ' + base64.b64encode(s).strip().decode('ascii')
                h.putheader('Authorization', s)
            h.endheaders()
            if self.method == "POST":
                h.send(data.encode('utf-8'))
            h.getresponse()    #can't do anything with the result
        except Exception:
            self.handleError(record)