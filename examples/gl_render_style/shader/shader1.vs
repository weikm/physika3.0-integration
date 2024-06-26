#version 330 core
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uvAttr;
layout (location = 3) in vec3 posAttr;
layout (location = 6) in float render_intensity;

out vec2 uv;
out vec4 eyeSpacePos;
out vec4 color;
out float gasOrFluid;
out vec3 particlePos;

uniform mat4 view;
uniform mat4 projection;

uniform sampler1D colorRamp;
uniform int mapping_fn;
uniform float radius;
uniform float max_value;
uniform float min_value;

float mapValue(float value, float min, float max, int fn){
    if(fn == 0)
        return 0.15f;
	if(fn == 1)
		return (value - min) / (max - min);
	if(fn == 2)
		return (sqrt(value) - sqrt(min)) / (sqrt(max) - sqrt(min));
	if(fn == 3)
		return (value * value - min * min) / (max * max - min * min);
	if(fn == 4)
		return (pow(value,1.f/3.f) - pow(min,1.f/3.f)) / (pow(max,1.f/3.f) - pow(min,1.f/3.f));
	if(fn == 5)
		return (value * value * value - min * min * min) / (max * max * max - min * min * min);
	if(fn == 6)
		return (log(value) - log(min)) / (log(max) - log(min));
	if(fn == 7)
	{
		if(value < min)
			return 0.8f;
		else if(value > max)
			return 0.2f;
	}
	if(fn == 8)
	{
		if(value < 0.2)
			return 0.2f;
		else if(value < 0.6)
			return 0.3f;
		else if(value < 1.1)
			return 0.5f;
		else if(value < 3.0)
			return 0.6f;
		else if(value < 5.0)
			return 0.8f;
		else
			return 1.0f;
	}
	return (value - min) / (max - min);
}

void main()
{
    // color
    float intensity = mapValue(render_intensity, min_value, max_value, mapping_fn);
    color = texture(colorRamp, intensity);
	
    // position
    uv = uvAttr;
	
    float V =  0.22 / radius;
	//float r = 1.0f;
	//if(posAttr.y < 0.3) r = 0.8f;
    //eyeSpacePos = view * vec4(r * position, 1.0);
	eyeSpacePos = view * vec4(position, 1.0);
    eyeSpacePos += vec4(posAttr * V * 2.0, 0.0);
    gl_Position = projection * eyeSpacePos;
	gasOrFluid = render_intensity;
	particlePos = posAttr;
}