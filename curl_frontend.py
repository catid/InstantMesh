from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import tempfile
import zipfile
import io
import argparse
import logging
from gradio_client import Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

logging.getLogger("httpx").setLevel(logging.WARNING)

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        client_address = self.client_address[0]
        logging.info(f"Received request from {client_address}")

        if self.headers["Authorization"] != args.auth_token:
            logging.warning(f"Unauthorized access attempt from {client_address}")
            self.send_response(401)
            self.end_headers()
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST"}
        )

        image_file = form["image"].file

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image_file.read())
            temp_file_path = temp_file.name

        client = Client(args.gradio_url)

        logging.info(f"Processing request from {client_address}")

        logging.info("Removing background...")
        background_removed_image = client.predict(
            temp_file_path,
            True,
            fn_index=2
        )

        logging.info("Generating views...")
        generated_views = client.predict(
            background_removed_image,
            40,
            5,
            fn_index=3
        )

        logging.info("Generating final results...")
        results = client.predict(
            fn_index=4
        )

        video = results[0]
        obj = results[1]
        glb = results[2]

        logging.info("Creating zip file...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            with open(video, "rb") as video_file:
                zip_file.writestr("video.mp4", video_file.read())

            with open(glb, "rb") as glb_file:
                zip_file.writestr("model.glb", glb_file.read())

        logging.info(f"Sending response to {client_address}")
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", "attachment; filename=results.zip")
        self.end_headers()

        zip_buffer.seek(0)
        self.wfile.write(zip_buffer.read())

        logging.info(f"Request from {client_address} completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP server with authentication")
    parser.add_argument("--auth-token", required=True, help="Authentication token")
    parser.add_argument("--port", type=int, default=8112, help="Port number (default: 8112)")
    parser.add_argument("--gradio-url", type=str, default="http://localhost:43839", help="URL to access")
    args = parser.parse_args()

    server_address = ("", args.port)
    httpd = HTTPServer(server_address, RequestHandler)
    logging.info(f"Server running on port {args.port}...")
    httpd.serve_forever()
