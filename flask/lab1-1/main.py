# Import dependencies
from flask import Flask
from flask_restful import Resource, Api, reqparse, abort
import json

# Configure server
app = Flask("VideoAPI")
api = Api(app)

# Define request parser
parser = reqparse.RequestParser()
parser.add_argument('title', required = True)
parser.add_argument('uploadDate', type=int, required = False)

# Load video data from file
with open('./videos.json','r') as f:
    videos = json.load(f)

# Write changes to file
def write_changes_to_file():
    global videos
    videos = { k: v for k, v in sorted(videos.items(),key=lambda video: video[1]["uploadDate"] or 0)}
    with open("./videos.json", 'w') as f:
        json.dump(videos, f)

# Define resources and endpoints

class Index(Resource):
    """
    Index resource
    """
    def get(self):
        return "Hello World!", 200

class AllVideos(Resource):
    """
    All videos resource
    """
    def get(self):
        return videos, 200

class VideoById(Resource):
    """
    Video by ID resource
    """
    def get(self, video_id):
        """
        Get video by ID
        """
        video = videos.get(video_id)
        if not video:
            abort(404, message="Video not found")
        return video, 200

    def put(self, video_id):
        """
        Update video by ID
        """
        video = videos.get(video_id)
        if not video:
            abort(404, message="Video not found")

        args = parser.parse_args()
        video['title'] = args['title']
        video['uploadDate'] = args['uploadDate']
        videos[video_id] = video
        write_changes_to_file()
        return video, 200

    def delete(self, video_id):
        """
        Delete video by ID
        """
        video = videos.pop(video_id, None)
        write_changes_to_file()
        if not video:
            abort(404, message="Video not found")
        return "", 204
    
class VideoSchedule(Resource):
    """
    Video schedule resource
    """
    def post(self):
        """
        Create a new video
        """
        args = parser.parse_args()
        video = {
            'title': args['title'],
            'uploadDate': args['uploadDate']
        }
        video_id = max([int(k.replace('video', '')) for k in videos.keys()]) + 1 if videos else 1
        videos[f'video{video_id}'] = video
        write_changes_to_file()
        return video, 201

# Add endpoints
api.add_resource(Index, "/")
api.add_resource(AllVideos, "/videos")
api.add_resource(VideoById, "/videos/<string:video_id>")
api.add_resource(VideoSchedule, "/videos")

# Start server
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8000)