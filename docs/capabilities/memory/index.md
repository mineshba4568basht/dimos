

```python

import pickle
from dimos.mapping.pointclouds.occupancy import general_occupancy, simple_occupancy, height_cost_occupancy
from dimos.mapping.occupancy.inflation import simple_inflate
from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.vis.color import color
from dimos.memory2.transform import downsample, throttle, speed
from dimos.memory2.vis.drawing.drawing import Drawing2D as Drawing
from dimos.utils.data import get_data
from dimos.memory2.vis.type import Point
from dimos.models.embedding.clip import CLIPModel
clip = CLIPModel()

store = SqliteStore(path=get_data("go2_bigoffice.db"))
global_map = pickle.loads(get_data("unitree_go2_bigoffice_map.pickle").read_bytes())

#costmap = simple_inflate(general_occupancy(global_map), 0.05)
#print("costmap", costmap)

drawing = Drawing()
drawing.add(global_map)



#store.streams.color_image.map(lambda obs: drawing.add(Point(obs.pose_stamped, color="#ff0000", radius=0.025))).last()

#store.streams.color_image_embedded.map(lambda obs: drawing.add(Point(obs.pose_stamped, color=color(obs.ts), radius=0.1))).drain()

#drawing.add(store.streams.color_image.transform(throttle(0.5)))


drawing.add(store.streams.color_image.transform(throttle(0.5)).transform(speed()))


embedded = store.streams.color_image_embedded
#bottle_pos = embedded.search(clip.embed_text("shop"), k=10)

#print(bottle_pos.summary())
#drawing.add(bottle_pos.map(lambda obs: obs.pose_stamped))
drawing.to_svg("assets/imageposes.svg")
```










![output](assets/imageposes.svg)
