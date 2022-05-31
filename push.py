from pushbullet import Pushbullet

API_KEY = "o.TyLaQgyHvpy4PoKM8TY2vesQDLJUYXHj"
filename = 'resolution.txt'

with open(filename, mode='r') as f:
    text = f.read()

pb = Pushbullet(API_KEY)
push = pb.push_note("This is the title", text)