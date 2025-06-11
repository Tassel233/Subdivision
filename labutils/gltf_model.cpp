#define TINYGLTF_IMPLEMENTATION      // 让 tinygltf 生成实现
#define STB_IMAGE_IMPLEMENTATION     // 如果没用 x-stb，则同时生成 stb
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "gltf_model.hpp"
#include <iostream>

using namespace labutils;

namespace
{

    bool readAccessor(const tinygltf::Model& model, int accessorIdx, std::vector<uint8_t>& out)
    {
        if (accessorIdx < 0) return false;

        const auto& accessor = model.accessors[accessorIdx];
        const auto& view = model.bufferViews[accessor.bufferView];
        const auto& buffer = model.buffers[view.buffer];

        size_t offset = view.byteOffset + accessor.byteOffset;
        size_t bytes = accessor.count *
            tinygltf::GetComponentSizeInBytes(accessor.componentType) *
            tinygltf::GetNumComponentsInType(accessor.type);

        out.assign(buffer.data.begin() + offset,
            buffer.data.begin() + offset + bytes);
        return true;
    }
}


bool GltfModel::loadFromFile(const std::string& path)
{
    tinygltf::TinyGLTF loader;
    tinygltf::Model    model;
    std::string warn, err;

    if (!loader.LoadASCIIFromFile(&model, &err, &warn, path)) {
        std::cerr << "[tinygltf] " << err << '\n';
        return false;
    }
    if (!warn.empty()) std::cerr << "[tinygltf] warn: " << warn << '\n';

    if (model.meshes.empty() || model.meshes.front().primitives.empty()) {
        std::cerr << "[gltf] no mesh data\n";
        return false;
    }
    const auto& prim = model.meshes.front().primitives.front();


    auto  getAttr = [&](const char* name) -> int {
        auto it = prim.attributes.find(name);
        return (it == prim.attributes.end()) ? -1 : it->second;
        };

    int posAcc = getAttr("POSITION");
    if (posAcc < 0) { std::cerr << "[gltf] POSITION missing\n"; return false; }

    int nrmAcc = getAttr("NORMAL");
    int uvAcc = getAttr("TEXCOORD_0");
    int idxAcc = prim.indices >= 0 ? prim.indices : -1;

    std::vector<uint8_t> posRaw, nrmRaw, uvRaw, idxRaw;
    readAccessor(model, posAcc, posRaw);
    readAccessor(model, nrmAcc, nrmRaw);
    readAccessor(model, uvAcc, uvRaw);
    readAccessor(model, idxAcc, idxRaw);

    size_t vtxCount = posRaw.size() / sizeof(glm::vec3);
    m_vertices.resize(vtxCount);
    m_indices.resize(idxRaw.size() / sizeof(uint32_t));

    std::memcpy(m_vertices.data(), posRaw.data(), posRaw.size());

    if (!nrmRaw.empty())
        std::memcpy(&m_vertices[0].normal, nrmRaw.data(), nrmRaw.size());
    else
        for (auto& v : m_vertices) v.normal = glm::vec3(0);

    if (!uvRaw.empty())
        std::memcpy(&m_vertices[0].uv, uvRaw.data(), uvRaw.size());
    else
        for (auto& v : m_vertices) v.uv = glm::vec2(0);

    if (!idxRaw.empty())
        std::memcpy(m_indices.data(), idxRaw.data(), idxRaw.size());
    else {
        m_indices.resize(vtxCount);
        std::iota(m_indices.begin(), m_indices.end(), 0u);
    }
    return true;
}