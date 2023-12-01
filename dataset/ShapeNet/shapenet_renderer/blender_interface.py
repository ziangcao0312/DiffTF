import os
import util
import bpy
import bmesh
import numpy as np

class BlenderInterface():
    def __init__(self, resolution=128, background_color=(1,1,1)):
        self.resolution = resolution

        # Delete the default cube (default selected)
        bpy.ops.object.delete()

        # Deselect all. All new object added to the scene will automatically selected.
        self.blender_renderer = bpy.context.scene.render
        self.blender_renderer.use_antialiasing = True
        self.blender_renderer.resolution_x = resolution
        self.blender_renderer.resolution_y = resolution
        self.blender_renderer.resolution_percentage = 100
        self.blender_renderer.image_settings.file_format = 'PNG'  # set output format to .png

        self.blender_renderer.alpha_mode = 'SKY'

        #################################################################################################
        # bpy.context.scene.use_nodes = True
        # bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
        # bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
        # self.nodes = bpy.context.scene.node_tree.nodes
        # for n in self.nodes:
        #     self.nodes.remove(n)
        # self.links = bpy.context.scene.node_tree.links

        # # Create depth output nodes
        # self.render_layers = self.nodes.new('CompositorNodeRLayers')
        # self.depth_file_output = self.nodes.new(type="CompositorNodeOutputFile")
        # self.depth_file_output.label = 'Depth Output'
        # self.depth_file_output.base_path = ''
        # self.depth_file_output.file_slots[0].use_node_format = True
        # self.depth_file_output.format.file_format = 'OPEN_EXR'
        # self.depth_file_output.format.color_depth = '16'
        # # if args.format == 'OPEN_EXR':
        # self.links.new(self.render_layers.outputs['Depth'], self.depth_file_output.inputs[0])
        # # self.depth_file_output.format.color_mode = "BW"
        # # # Remap as other types can not represent the full range of depth.
        # # self.map = self.nodes.new(type="CompositorNodeMapValue")
        # # # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        # # self.map.offset = [-0.7]
        # # self.map.size = [1.4]
        # # self.map.use_min = True
        # # self.map.min = [0]
        # # self.links.new(self.render_layers.outputs['Depth'], self.map.inputs[0])
        # # self.links.new(self.map.outputs[0], self.depth_file_output.inputs[0])

        # # Create normal output nodes
        # self.scale_node = self.nodes.new(type="CompositorNodeMixRGB")
        # self.scale_node.blend_type = 'MULTIPLY'
        # # self.scale_node.use_alpha = True
        # self.scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        # self.links.new(self.render_layers.outputs['Normal'], self.scale_node.inputs[1])

        # self.bias_node = self.nodes.new(type="CompositorNodeMixRGB")
        # self.bias_node.blend_type = 'ADD'
        # # self.bias_node.use_alpha = True
        # self.bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        # self.links.new(self.scale_node.outputs[0], self.bias_node.inputs[1])

        # self.normal_file_output = self.nodes.new(type="CompositorNodeOutputFile")
        # self.normal_file_output.label = 'Normal Output'
        # self.normal_file_output.base_path = ''
        # self.normal_file_output.file_slots[0].use_node_format = True
        # self.normal_file_output.format.file_format = 'PNG'
        # # self.links.new(self.render_layers.outputs['Normal'], self.normal_file_output.inputs[0])
        # self.links.new(self.bias_node.outputs[0], self.normal_file_output.inputs[0])
        ##############################################################################################################

        world = bpy.context.scene.world
        world.horizon_color = background_color
        world.light_settings.use_environment_light = True
        world.light_settings.environment_color = 'SKY_COLOR'
        world.light_settings.environment_energy = 1.

        lamp1 = bpy.data.lamps['Lamp']
        lamp1.type = 'SUN'
        lamp1.shadow_method = 'NOSHADOW'
        lamp1.use_specular = False
        lamp1.energy = 0.5
        bpy.data.objects['Lamp'].rotation_euler[0] = bpy.data.objects['Lamp'].rotation_euler[1] = bpy.data.objects['Lamp'].rotation_euler[2] = 0
        bpy.data.objects['Lamp'].location[2] = 10

        bpy.ops.object.lamp_add(type='SUN')
        lamp2 = bpy.data.lamps['Sun']
        lamp2.shadow_method = 'NOSHADOW'
        lamp2.use_specular = False
        lamp2.energy = 0.5
        bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
        bpy.data.objects['Sun'].rotation_euler[0] += 180
        bpy.data.objects['Sun'].location[2] = -10
        
        bpy.ops.object.lamp_add(type='SUN')
        lamp2 = bpy.data.lamps['Sun.001']
        lamp2.shadow_method = 'NOSHADOW'
        lamp2.use_specular = False
        lamp2.energy = 0.5
        bpy.data.objects['Sun.001'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
        bpy.data.objects['Sun.001'].rotation_euler[0] += 90
        bpy.data.objects['Sun.001'].location[1] = -10

        bpy.ops.object.lamp_add(type='SUN')
        lamp2 = bpy.data.lamps['Sun.002']
        lamp2.shadow_method = 'NOSHADOW'
        lamp2.use_specular = False
        lamp2.energy = 0.5
        bpy.data.objects['Sun.002'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
        bpy.data.objects['Sun.002'].rotation_euler[0] += 270
        bpy.data.objects['Sun.002'].location[1] = 10

        bpy.ops.object.lamp_add(type='SUN')
        lamp2 = bpy.data.lamps['Sun.003']
        lamp2.shadow_method = 'NOSHADOW'
        lamp2.use_specular = False
        lamp2.energy = 0.5
        bpy.data.objects['Sun.003'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
        bpy.data.objects['Sun.003'].rotation_euler[1] += 90
        bpy.data.objects['Sun.003'].location[0] = 10

        bpy.ops.object.lamp_add(type='SUN')
        lamp2 = bpy.data.lamps['Sun.004']
        lamp2.shadow_method = 'NOSHADOW'
        lamp2.use_specular = False
        lamp2.energy = 0.5
        bpy.data.objects['Sun.004'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
        bpy.data.objects['Sun.004'].rotation_euler[1] -= 90
        bpy.data.objects['Sun.004'].location[0] = -10


        # Set up the camera
        self.camera = bpy.context.scene.camera
        self.camera.data.sensor_height = self.camera.data.sensor_width # Square sensor
        util.set_camera_focal_length_in_world_units(self.camera.data, 525./512*resolution) # Set focal length to a common value (kinect)

        bpy.ops.object.select_all(action='DESELECT')


    def cleanup(self, obj):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.context.scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
        bpy.ops.object.mode_set(mode='OBJECT')


    def import_mesh(self, fpath, scale=1., object_world_matrix=None):
        ext = os.path.splitext(fpath)[-1]
        if ext == '.obj':
            bpy.ops.import_scene.obj(filepath=str(fpath), split_mode='OFF')
        elif ext == '.ply':
            bpy.ops.import_mesh.ply(filepath=str(fpath))

        obj = bpy.context.selected_objects[0]

        # obj.select = True
        # bpy.ops.object.shade_smooth()
        # Add a normal display modifier to the object
        # mod = obj.modifiers.new(name="Normal Display", type='NORMAL_EDIT')
        # mod.show_on_cage = True
        # mod.use_color_ramp = True
        # mod.use_stretch = True
        # mod.object_space = True

        # util.dump(bpy.context.selected_objects)

        self.cleanup(obj)

        if object_world_matrix is not None:
            obj.matrix_world = object_world_matrix

        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0., 0., 0.) # center the bounding box!

        verts = np.array([vert.co.to_tuple() for vert in obj.data.vertices])
        minv = verts.min(0)
        maxv = verts.max(0)
        center = (minv + maxv) / 2
        length = np.sqrt(((maxv - minv) ** 2).sum())
        scale = scale / length

        obj.location = (center[0], center[1], center[2])

        print(scale, center, length, minv, maxv)

        if scale != 1.:
            bpy.ops.transform.resize(value=(scale, scale, scale))

        # Disable transparency & specularities
        M = bpy.data.materials
        for i in range(len(M)):
            M[i].use_transparency = False
            M[i].specular_intensity = 0.0

        # Disable texture interpolation
        T = bpy.data.textures
        for i in range(len(T)):
            try:
                T[i].use_interpolation = False
                T[i].use_mipmap = False
                T[i].use_filter_size_min = True
                T[i].filter_type = "BOX"
            except:
                continue

    def render(self, output_dir, blender_cam2world_matrices, write_cam_params=False):

        if write_cam_params:
            img_dir = os.path.join(output_dir, 'rgb')
            pose_dir = os.path.join(output_dir, 'pose')

            util.cond_mkdir(img_dir)
            util.cond_mkdir(pose_dir)
        else:
            img_dir = output_dir
            util.cond_mkdir(img_dir)

        if write_cam_params:
            K = util.get_calibration_matrix_K_from_blender(self.camera.data)
            print(K)
            with open(os.path.join(output_dir, 'intrinsics.txt'),'w') as intrinsics_file:
                intrinsics_file.write('%f %f %f 0.\n'%(K[0][0], K[0][2], K[1][2]))
                intrinsics_file.write('0. 0. 0.\n')
                intrinsics_file.write('1.\n')
                intrinsics_file.write('%d %d\n'%(self.resolution, self.resolution))

        for i in range(len(blender_cam2world_matrices)):
            self.camera.matrix_world = blender_cam2world_matrices[i]

            # Render the object
            if os.path.exists(os.path.join(img_dir, '%06d.png' % i)):
                continue

            # Render the color image
            self.blender_renderer.filepath = os.path.join(img_dir, '%06d.png'%i)
            # self.depth_file_output.file_slots[0].path = os.path.join(img_dir, '%06d_depth'%i)
            # self.normal_file_output.file_slots[0].path = os.path.join(img_dir, '%06d_normal'%i)
            bpy.ops.render.render(write_still=True)

            if write_cam_params:
                # Write out camera pose
                RT = util.get_world2cam_from_blender_cam(self.camera)
                cam2world = RT.inverted()
                with open(os.path.join(pose_dir, '%06d.txt'%i),'w') as pose_file:
                    matrix_flat = []
                    for j in range(4):
                        for k in range(4):
                            matrix_flat.append(cam2world[j][k])
                    pose_file.write(' '.join(map(str, matrix_flat)) + '\n')

        # Remember which meshes were just imported
        meshes_to_remove = []
        for ob in bpy.context.selected_objects:
            meshes_to_remove.append(ob.data)

        bpy.ops.object.delete()

        # Remove the meshes from memory too
        for mesh in meshes_to_remove:
            bpy.data.meshes.remove(mesh)