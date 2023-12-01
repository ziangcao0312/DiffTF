## ShapeNet
1. Download the ShapeNet data from this

[website]: https://shapenet.org/

2. Rendering the multi-view images using this

[repo]: https://github.com/vsitzmann/shapenet_renderer/tree/master

```
bash render.sh   #rendering a serial of objects
```

We give a simple example in this folder

```
renders
--ShapeNet_Car
----Car_1_name 
------pose
--------000000.txt
--------000001.txt
…………
--------000194.txt   # determined by the number of rendering images
------rgb
--------000000.png
--------000001.png
…………
--------000194.png   # determined by the number of rendering images
------intrinsics.txt
----Car_2_name
…………
----Car_n_name
--ShapeNet_plane
--ShapeNet_chair
```

