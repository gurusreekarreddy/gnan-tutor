from openenv.server import create_web_interface_app
from server.gnan_tutor_environment import GnanTutorEnv

app = create_web_interface_app(GnanTutorEnv)
