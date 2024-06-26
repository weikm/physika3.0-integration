#version 330 core
out vec4 FragColor;
in vec2 uv;
in vec4 eyeSpacePos;
in vec4 color;
in float gasOrFluid;
in vec3 particlePos;
uniform mat4 projection;

void main()
{
    vec3 N;
    N.xy = uv.xy * vec2(2.0, 2.0) - vec2(1.0, 1.0);
    float r2 = dot(N.xy, N.xy);
    if( r2 > 1.0f) discard;
    float alpha_v = 0.2f;

    N.z = sqrt(1.0f - r2);

    vec3 lightDir = vec3(0, 0, 1);
    float diffuse = abs(dot(N, lightDir));
    FragColor = vec4(diffuse, diffuse, diffuse, 1.0f) * color;
    vec4 sphereEyeSpacePos;
    sphereEyeSpacePos.xyz = eyeSpacePos.xyz + N * eyeSpacePos.w;
    sphereEyeSpacePos.w = 1.0;
    vec4 projPos = (projection * sphereEyeSpacePos);
    gl_FragDepth = (projPos.z / projPos.w) * 0.5 + 0.5;
}