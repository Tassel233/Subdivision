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


//bool GltfModel::loadFromFile(const std::string& path)
//{
//    tinygltf::TinyGLTF loader;
//    tinygltf::Model    model;
//    std::string warn, err;
//
//    if (!loader.LoadASCIIFromFile(&model, &err, &warn, path)) {
//        std::cerr << "[tinygltf] " << err << '\n';
//        return false;
//    }
//    if (!warn.empty()) std::cerr << "[tinygltf] warn: " << warn << '\n';
//
//    if (model.meshes.empty() || model.meshes.front().primitives.empty()) {
//        std::cerr << "[gltf] no mesh data\n";
//        return false;
//    }
//    const auto& prim = model.meshes.front().primitives.front();
//
//
//    auto  getAttr = [&](const char* name) -> int {
//        auto it = prim.attributes.find(name);
//        return (it == prim.attributes.end()) ? -1 : it->second;
//        };
//
//    int posAcc = getAttr("POSITION");
//    if (posAcc < 0) { std::cerr << "[gltf] POSITION missing\n"; return false; }
//
//    int nrmAcc = getAttr("NORMAL");
//    int uvAcc = getAttr("TEXCOORD_0");
//    int idxAcc = prim.indices >= 0 ? prim.indices : -1;
//
//    std::vector<uint8_t> posRaw, nrmRaw, uvRaw, idxRaw;
//    readAccessor(model, posAcc, posRaw);
//    readAccessor(model, nrmAcc, nrmRaw);
//    readAccessor(model, uvAcc, uvRaw);
//    readAccessor(model, idxAcc, idxRaw);
//
//    size_t vtxCount = posRaw.size() / sizeof(glm::vec3);
//    m_vertices.resize(vtxCount);
//    m_indices.resize(idxRaw.size() / sizeof(uint32_t));
//
//    std::memcpy(m_vertices.data(), posRaw.data(), posRaw.size());
//
//    if (!nrmRaw.empty())
//        std::memcpy(&m_vertices[0].normal, nrmRaw.data(), nrmRaw.size());
//    else
//        for (auto& v : m_vertices) v.normal = glm::vec3(0);
//
//    if (!uvRaw.empty())
//        std::memcpy(&m_vertices[0].uv, uvRaw.data(), uvRaw.size());
//    else
//        for (auto& v : m_vertices) v.uv = glm::vec2(0);
//
//    if (!idxRaw.empty())
//        std::memcpy(m_indices.data(), idxRaw.data(), idxRaw.size());
//    else {
//        m_indices.resize(vtxCount);
//        std::iota(m_indices.begin(), m_indices.end(), 0u);
//    }
//    return true;
//}

bool GltfModel::loadFromFile(const std::string& path) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
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

    auto getAttr = [&](const char* name) -> int {
        auto it = prim.attributes.find(name);
        return (it == prim.attributes.end()) ? -1 : it->second;
        };

    int posAcc = getAttr("POSITION");
    if (posAcc < 0) {
        std::cerr << "[gltf] POSITION missing\n";
        return false;
    }

    // 验证位置属性
    const auto& posAccessor = model.accessors[posAcc];
    if (posAccessor.type != TINYGLTF_TYPE_VEC3 ||
        posAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        std::cerr << "[gltf] Invalid POSITION attribute type\n";
        return false;
    }

    int nrmAcc = getAttr("NORMAL");
    int uvAcc = getAttr("TEXCOORD_0");
    int idxAcc = prim.indices >= 0 ? prim.indices : -1;

    // 读取数据
    std::vector<uint8_t> posRaw;
    if (!readAccessor(model, posAcc, posRaw)) {
        std::cerr << "[gltf] Failed to read POSITION data\n";
        return false;
    }

    size_t vtxCount = posAccessor.count;
    m_vertices.resize(vtxCount);

    // 处理位置数据
    const auto* posData = reinterpret_cast<const glm::vec3*>(posRaw.data());
    for (size_t i = 0; i < vtxCount; ++i) {
        m_vertices[i].pos = posData[i];
    }

    // 处理法线数据
    if (nrmAcc >= 0) {
        const auto& nrmAccessor = model.accessors[nrmAcc];
        if (nrmAccessor.type == TINYGLTF_TYPE_VEC3 &&
            nrmAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
        {
            std::vector<uint8_t> nrmRaw;
            if (readAccessor(model, nrmAcc, nrmRaw) &&
                nrmRaw.size() == vtxCount * sizeof(glm::vec3))
            {
                const auto* nrmData = reinterpret_cast<const glm::vec3*>(nrmRaw.data());
                for (size_t i = 0; i < vtxCount; ++i) {
                    m_vertices[i].normal = nrmData[i];
                }
            }
            else {
                std::cerr << "[gltf] Failed to read or invalid NORMAL data\n";
            }
        }
        else {
            std::cerr << "[gltf] Invalid NORMAL attribute type\n";
        }
    }
    else {
        for (auto& v : m_vertices) v.normal = glm::vec3(0, 1, 0); // 默认朝上
    }

    // 处理UV数据（类似法线）
    // ...

    // 处理索引数据
    if (idxAcc >= 0) {
        const auto& idxAccessor = model.accessors[idxAcc];
        const auto& view = model.bufferViews[idxAccessor.bufferView];
        const auto& buffer = model.buffers[view.buffer];
        const uint8_t* data = buffer.data.data() + view.byteOffset + idxAccessor.byteOffset;

        m_indices.resize(idxAccessor.count);

        switch (idxAccessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            std::memcpy(m_indices.data(), data, idxAccessor.count * sizeof(uint32_t));
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            for (size_t i = 0; i < idxAccessor.count; ++i) {
                m_indices[i] = reinterpret_cast<const uint16_t*>(data)[i];
            }
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            for (size_t i = 0; i < idxAccessor.count; ++i) {
                m_indices[i] = reinterpret_cast<const uint8_t*>(data)[i];
            }
            break;
        default:
            std::cerr << "Unsupported index type: " << idxAccessor.componentType << "\n";
            return false;
        }
    }
    else {
        m_indices.resize(vtxCount);
        std::iota(m_indices.begin(), m_indices.end(), 0u);
    }

    return true;
}