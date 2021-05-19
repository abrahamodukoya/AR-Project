import math

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
# from OpenGL.raw.GL.VERSION.GL_2_0 import glUniformMatrix4fv
from PIL import Image
import numpy as np
import assimp_py as asspy
from functools import reduce
import ctypes
from constants import CHARS

null = ctypes.c_void_p(0)

char_models = []
curr_model = None
curr_img = None
curr_camera_model = np.identity(4)


# column-major order
def perspective(fov, aspect, near, far):
    mat = np.zeros((4, 4))
    t = math.tan(math.radians(fov * 0.5)) * near
    b = -t
    le = aspect * b
    r = aspect * t
    mat[0, 0] = (2 * near) / (r - le)
    mat[1, 1] = (2 * near) / (t - b)
    mat[2, 2] = far / (far - near)
    mat[2, 3] = (near * far) / (near - far)
    mat[3, 2] = 1
    return np.transpose(mat, (1, 0))


class Shader:
    def __init__(self, shader_file, shader_type):
        self.shader_id = glCreateShader(shader_type)
        self.shader_type = shader_type
        with open(shader_file, 'rb') as f:
            self.shader_text = f.read()

    def bind_and_compile(self):
        glShaderSource(self.shader_id, [self.shader_text])
        glCompileShader(self.shader_id)
        success = glGetShaderiv(self.shader_id, GL_COMPILE_STATUS)
        if success == GL_FALSE:
            info_log = glGetShaderInfoLog(self.shader_id)
            print(
                f'Error compiling {"vertex" if self.shader_type == GL_VERTEX_SHADER else "fragment"} shader -> info log: {info_log}')
            return False
        return True


class ShaderProgram:
    def __init__(self, shader_program_id):
        if shader_program_id == 0:
            print('Error creating shader program')
            exit(1)
        self.shader_program_id = shader_program_id
        self.vertex_shader = None
        self.fragment_shader = None

    def set_vertex_shader(self, vertex_shader):
        self.vertex_shader = vertex_shader
        glAttachShader(self.shader_program_id, self.vertex_shader.shader_id)

    def set_fragment_shader(self, fragment_shader):
        self.fragment_shader = fragment_shader
        glAttachShader(self.shader_program_id, self.fragment_shader.shader_id)

    def link_program(self, vpos_index=7, vtex_index=9):
        glBindAttribLocation(self.shader_program_id, vpos_index, 'vertex_position')
        # glBindAttribLocation(self.shader_program_id, 8, 'vertex_normal')
        glBindAttribLocation(self.shader_program_id, vtex_index, 'vertex_texture')
        glBindFragDataLocation(self.shader_program_id, 0, "fragColor")
        glLinkProgram(self.shader_program_id)
        success = glGetProgramiv(self.shader_program_id, GL_LINK_STATUS)
        if success == GL_FALSE:
            print('Error linking shader program')
            return False

        glValidateProgram(self.shader_program_id)
        success = glGetProgramiv(self.shader_program_id, GL_VALIDATE_STATUS)
        if success == GL_FALSE:
            print('Invalid shader program')
            return False
        return True

    def use_program(self):
        glUseProgram(self.shader_program_id)


class ModelObject:
    def __init__(self):
        self.model = np.identity(4)
        self.view = np.identity(4)
        self.proj = np.identity(4)

        self.vao = None
        self.vertex_vbo = None
        self.normal_vbo = None
        self.texture_vbo = None

        self.texture_image = None
        self.tex = None
        self.texture_width = None
        self.texture_height = None

        self.shader_program = None
        self.num_points = None

    def populate_vbo(self, data, vbo_id):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        data = reduce(lambda a, b: a + list(b), data, [])

        data_type = GLfloat * len(data)
        size_of_float = ctypes.sizeof(GLfloat)

        glBufferData(GL_ARRAY_BUFFER, len(data) * size_of_float, data_type(*data), GL_STATIC_DRAW)

    def initialise_vao(self):
        self.vao = glGenVertexArrays(1)

    def initialise_tex(self, file):
        glEnable(GL_TEXTURE_2D)
        self.tex = glGenTextures(1)
        texture_jpg = Image.open(file)
        self.texture_width, self.texture_height = texture_jpg.size
        self.texture_image = texture_jpg.tobytes('raw', 'RGB', 0, -1)

    def set_up_texture(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.texture_width, self.texture_height, 0, GL_RGB,
                     GL_UNSIGNED_BYTE, self.texture_image)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glGenerateMipmap(GL_TEXTURE_2D)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    def populate_vertex_vbo(self, data):
        self.vertex_vbo = glGenBuffers(1)
        self.num_points = len(data)
        self.populate_vbo(data, self.vertex_vbo)

    def populate_normal_vbo(self, data):
        self.normal_vbo = glGenBuffers(1)
        self.populate_vbo(data, self.normal_vbo)

    def populate_texture_vbo(self, data):
        self.texture_vbo = glGenBuffers(1)
        self.populate_vbo(data, self.texture_vbo)

    def link_vbo_to_shaders(self):
        pos = glGetAttribLocation(self.shader_program.shader_program_id, "vertex_position")
        # norm = glGetAttribLocation(self.shader_program.shader_program_id, "vertex_normal")
        texture = glGetAttribLocation(self.shader_program.shader_program_id, "vertex_texture")

        glBindVertexArray(self.vao)

        glEnableVertexAttribArray(pos)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbo)
        glVertexAttribPointer(pos, 3, GL_FLOAT, GL_FALSE, 0, null)
        # glEnableVertexAttribArray(norm)
        # glBindBuffer(GL_ARRAY_BUFFER, self.normal_vbo)
        # glVertexAttribPointer(norm, 3, GL_FLOAT, GL_FALSE, 0, 0)

        glEnableVertexAttribArray(texture)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_vbo)
        glVertexAttribPointer(texture, 2, GL_FLOAT, GL_FALSE, 0, null)

    def draw(self):
        self.model = curr_camera_model
        # self.proj = perspective(33.7, (640 / 480), 0.1, 1000)
        glBindVertexArray(self.vao)
        sp_id = self.shader_program.shader_program_id
        self.set_up_texture()
        # glActiveTexture(GL_TEXTURE0)
        # glBindTexture(GL_TEXTURE_2D, self.tex)
        matrix_location = glGetUniformLocation(sp_id, "model")
        view_mat_location = glGetUniformLocation(sp_id, "view")
        proj_mat_location = glGetUniformLocation(sp_id, "proj")
        tex_location = glGetUniformLocation(sp_id, "tex")
        glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, self.proj)
        glUniformMatrix4fv(view_mat_location, 1, GL_FALSE, self.view)
        glUniformMatrix4fv(matrix_location, 1, GL_FALSE, self.model)
        glUniform1i(tex_location, 0)
        glDrawArrays(GL_TRIANGLES, 0, self.num_points)

    def use(self):
        self.shader_program.use_program()


class ImgObject(ModelObject):
    def populate_vbo(self, data, vbo_id):
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        data_type = GLfloat * len(data)
        size_of_float = ctypes.sizeof(GLfloat)
        glBufferData(GL_ARRAY_BUFFER, len(data) * size_of_float, data_type(*data), GL_STATIC_DRAW)

    def initialise_tex(self, file=None):
        glEnable(GL_TEXTURE_2D)
        self.tex = glGenTextures(1)

    def set_up_texture(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0, GL_BGR, GL_UNSIGNED_BYTE, self.texture_image)
        # glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # glTexEnv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        # Parameteri -> Parameterf
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        # glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        # glEnable(GL_TEXTURE_2D)
        # glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, 3, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE,
        #             self.texture_image)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        # img_sampler = glGetUniformLocation(self.shader_program.shader_program_id, 'img')
        # glUniform1i(img_sampler, 0)

    def draw(self):
        glLoadIdentity()
        glBindVertexArray(self.vao)
        self.set_up_texture()
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glPushMatrix()
        glTranslatef(0, 0, -10)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(4.0, 3.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(-4.0, 3.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(-4.0, -3.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(4.0, -3.0, 0.0)
        glEnd()
        glPopMatrix()

        # self.model[-1, 2] = -10
        # self.proj = perspective(33.7, (640 / 480), 0.1, 100)

        # sp_id = self.shader_program.shader_program_id
        # matrix_location = glGetUniformLocation(sp_id, "model")
        # view_mat_location = glGetUniformLocation(sp_id, "view")
        # proj_mat_location = glGetUniformLocation(sp_id, "proj")
        # tex_location = glGetUniformLocation(sp_id, "tex")
        # glUniformMatrix4fv(proj_mat_location, 1, GL_FALSE, self.proj)
        # glUniformMatrix4fv(view_mat_location, 1, GL_FALSE, self.view)
        # glUniformMatrix4fv(matrix_location, 1, GL_FALSE, self.model)
        # glUniform1i(tex_location, 0)
        # glDrawArrays(GL_QUADS, 0, 12)

    # def link_vbo_to_shaders(self):
    #     a_pos = glGetAttribLocation(self.shader_program.shader_program_id, "aPos")  # 3
    #     glBindBuffer(GL_ARRAY_BUFFER, self.vertex_vbo)
    #     glBindVertexArray(self.vao)
    #     glEnableVertexAttribArray(a_pos)
    #     glVertexAttribPointer(a_pos, 3, GL_FLOAT, False, 0, 0)


def display():  # , img_shader_program, img_vao, img_vbo, img_texture):
    # glDisable(GL_DEPTH_TEST)
    # glEnable(GL_DEPTH_TEST)
    # glDepthFunc(GL_LESS)
    glClearColor(0, 0, 0, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(33.7, (640 / 480), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # image background
    # glDepthFunc(GL_LEQUAL)
    # glLoadIdentity()
    # glEnable(GL_TEXTURE_2D)
    if curr_img is not None:
        # print('should be drawing the camera frame')
        # curr_img.use()
        glUseProgram(0)
        curr_img.draw()


    # glEnable(GL_LIGHTING)

    if curr_model is not None:
        # print('should be drawing a model')
        glEnable(GL_DEPTH_TEST)
        curr_model.use()
        curr_model.draw()

    glutSwapBuffers()


def init():
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    shader_program = ShaderProgram(glCreateProgram())
    vertex_shader = Shader('shaders/vertexShader.txt', GL_VERTEX_SHADER)
    fragment_shader = Shader('shaders/fragmentShader.txt', GL_FRAGMENT_SHADER)
    if not vertex_shader.bind_and_compile() or not fragment_shader.bind_and_compile():
        exit(1)
    shader_program.set_vertex_shader(vertex_shader)
    shader_program.set_fragment_shader(fragment_shader)
    if not shader_program.link_program():
        exit(1)

    for char in CHARS:
        model_data = asspy.ImportFile(f'3d_models/{char}.obj', asspy.Process_Triangulate)
        model = ModelObject()
        model.initialise_vao()
        model.initialise_tex(f'textures/{char}_texture.jpg')
        model.populate_vertex_vbo(model_data.meshes[0].vertices)
        model.populate_normal_vbo(model_data.meshes[0].normals)
        model.populate_texture_vbo(model_data.meshes[0].texcoords[0])
        model.shader_program = shader_program
        model.link_vbo_to_shaders()
        char_models.append(model)

    global curr_img
    img_shader_program = ShaderProgram(glCreateProgram())
    img_vertex_shader = Shader('shaders/vertexShader.txt', GL_VERTEX_SHADER)
    img_fragment_shader = Shader('shaders/fragmentShader.txt', GL_FRAGMENT_SHADER)
    if not img_vertex_shader.bind_and_compile() or not img_fragment_shader.bind_and_compile():
        exit(1)
    img_shader_program.set_vertex_shader(img_vertex_shader)
    img_shader_program.set_fragment_shader(img_fragment_shader)
    if not img_shader_program.link_program(vpos_index=8, vtex_index=10):
        exit(1)

    curr_img = ImgObject()
    curr_img.initialise_vao()
    curr_img.initialise_tex()
    curr_img.populate_vertex_vbo([-4.0, 3.0, 0.0, 4.0, 3.0, 0.0, 4.0, -3.0, 0.0, -4.0, -3.0, 0.0])
    curr_img.populate_texture_vbo([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    curr_img.shader_program = img_shader_program
    curr_img.link_vbo_to_shaders()
    # placeholder for first call to display
    placeholder = Image.open('textures/placeholder.jpg')
    curr_img.texture_image = placeholder.tobytes('raw', 'RGB', 0, -1)


def launch(update_scene_func):
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(640, 480)
    glutCreateWindow('AR Project')
    glutDisplayFunc(display)
    glutIdleFunc(update_scene_func)
    init()
    glutMainLoop()
