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

    struct Edge {
        uint32_t a, b;
        Edge(uint32_t x, uint32_t y) : a(std::min(x, y)), b(std::max(x, y)) {}
        bool operator<(const Edge& rhs) const {
            return std::tie(a, b) < std::tie(rhs.a, rhs.b);
        }
    };

    void triangle_to_quads(
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& indices,
        std::vector<glm::vec3>& out_positions,
        std::vector<glm::uvec4>& out_quads)
    {
        std::map<Edge, uint32_t> edgePoints;
        out_positions = positions;

        for (size_t i = 0; i < indices.size(); i += 3)
        {
            uint32_t a = indices[i + 0];
            uint32_t b = indices[i + 1];
            uint32_t c = indices[i + 2];

            glm::vec3 fa = positions[a];
            glm::vec3 fb = positions[b];
            glm::vec3 fc = positions[c];

            // Face center point
            glm::vec3 face_center = (fa + fb + fc) / 3.f;
            uint32_t fidx = static_cast<uint32_t>(out_positions.size());
            out_positions.push_back(face_center);

            // Edge midpoints
            auto get_mid = [&](uint32_t u, uint32_t v) {
                Edge e(u, v);
                if (edgePoints.count(e)) return edgePoints[e];
                glm::vec3 midpoint = (positions[u] + positions[v]) * 0.5f;
                uint32_t idx = static_cast<uint32_t>(out_positions.size());
                out_positions.push_back(midpoint);
                edgePoints[e] = idx;
                return idx;
                };

            uint32_t ab = get_mid(a, b);
            uint32_t bc = get_mid(b, c);
            uint32_t ca = get_mid(c, a);

            // 3 quads per triangle
            out_quads.emplace_back(glm::uvec4{ a, ab, fidx, ca });
            out_quads.emplace_back(glm::uvec4{ b, bc, fidx, ab });
            out_quads.emplace_back(glm::uvec4{ c, ca, fidx, bc });
        }
    }

}

void GltfModel::generateTrianglesFromQuads()
{
    m_quadIndices.clear();
    m_quadIndices.reserve(m_quadFaces.size() * 6); // 每个 quad 2 三角形 = 6 indices

    for (const auto& q : m_quadFaces)
    {
        // Triangle 1: a, b, c
        m_quadIndices.push_back(q.x);
        m_quadIndices.push_back(q.y);
        m_quadIndices.push_back(q.z);

        // Triangle 2: a, c, d
        m_quadIndices.push_back(q.x);
        m_quadIndices.push_back(q.z);
        m_quadIndices.push_back(q.w);
    }
}


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

    // convert from triangle to quad to fill m_quadFaces
    // Step 1: 提取原始 position
    std::vector<glm::vec3> rawPos;
    for (auto& v : m_vertices)
        rawPos.push_back(v.pos);

    // Step 2: 生成 quad 顶点坐标 + quad 索引
    std::vector<glm::vec3> quadPos;
    triangle_to_quads(rawPos, m_indices, quadPos, m_quadFaces);

    // Step 3: 构建 m_quadVertices（带默认 normal/uv）
    m_quadVertices.resize(quadPos.size());
    for (size_t i = 0; i < quadPos.size(); ++i) {
        m_quadVertices[i].pos = quadPos[i];
        m_quadVertices[i].normal = glm::vec3(0, 1, 0); // 默认法线
        m_quadVertices[i].uv = glm::vec2(0);           // 默认 UV
    }
    generateTrianglesFromQuads();

    return true;
}