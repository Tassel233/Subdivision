#define TINYGLTF_IMPLEMENTATION 
#define STB_IMAGE_IMPLEMENTATION 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#include <glm/gtc/epsilon.hpp>
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

    // 把几何上相同(ε)的位置合并，更新 indices
    template<typename TVertex>
    void weldVertices(std::vector<TVertex>& verts,
        std::vector<uint32_t>& indices,
        float eps = 1e-5f)
    {
        std::vector<TVertex>   unique;
        std::vector<uint32_t>  remap(verts.size());

        for (size_t i = 0; i < verts.size(); ++i)
        {
            const glm::vec3& p = verts[i].pos;

            uint32_t hit = UINT32_MAX;
            for (uint32_t j = 0; j < unique.size(); ++j)
                if (glm::all(glm::epsilonEqual(p, unique[j].pos, eps)))
                {
                    hit = j; break;
                }

            if (hit == UINT32_MAX) {         // 第一次见到这个坐标
                hit = static_cast<uint32_t>(unique.size());
                unique.push_back(verts[i]);
            }
            remap[i] = hit;
        }

        for (auto& id : indices) id = remap[id];  // 重映射索引
        verts.swap(unique);                       // 压缩顶点表
    }

}

void GltfModel::generateTrianglesFromQuads()
{
    m_quadIndicesRaw.clear();
    m_quadIndicesRaw.reserve(m_quadFacesRaw.size() * 6); // 每个 quad 2 三角形 = 6 indices

    for (const auto& q : m_quadFacesRaw)
    {
        // Triangle 1: a, b, c
        m_quadIndicesRaw.push_back(q.x);
        m_quadIndicesRaw.push_back(q.y);
        m_quadIndicesRaw.push_back(q.z);

        // Triangle 2: a, c, d
        m_quadIndicesRaw.push_back(q.x);
        m_quadIndicesRaw.push_back(q.z);
        m_quadIndicesRaw.push_back(q.w);
    }
}


void GltfModel::preprocessForSubdivision() {
    // 建立 position → canonical index
    std::unordered_map<glm::vec3, uint32_t, PosHasher> canonicalPosMap;
    std::vector<uint32_t> vertexRemap(m_quadVerticesRaw.size());

    for (uint32_t i = 0; i < m_quadVerticesRaw.size(); ++i) {
        glm::vec3 pos = m_quadVerticesRaw[i].pos;
        bool found = false;
        for (const auto& [p, idx] : canonicalPosMap) {
            if (pos_equal(p, pos)) {
                vertexRemap[i] = idx;
                found = true;
                break;
            }
        }
        if (!found) {
            vertexRemap[i] = i;
            canonicalPosMap[pos] = i;
        }
    }


    std::map<Edge, std::vector<uint32_t>> edgeFaces;  // Edge -> Faces
    std::map<uint32_t, std::vector<uint32_t>> vertexFaces; // Vertex -> Faces
    std::map<uint32_t, std::vector<uint32_t>> vertexEdges; // Vertex -> Edges

    // Step 1: 构建 edgeList 与 edgeToFace 映射
    std::map<Edge, uint32_t> edgeIndexMap;
    for (uint32_t faceID = 0; faceID < m_quadFacesRaw.size(); ++faceID) {
        const glm::uvec4& q = m_quadFacesRaw[faceID];
        uint32_t vs[4] = { q.x, q.y, q.z, q.w };
        for (int i = 0; i < 4; ++i) {
            //Edge e(vs[i], vs[(i + 1) % 4]);
            Edge e(vertexRemap[vs[i]], vertexRemap[vs[(i + 1) % 4]]);
            uint32_t eid;
            if (edgeIndexMap.count(e) == 0) {
                eid = static_cast<uint32_t>(m_edgeList.size());
                edgeIndexMap[e] = eid;
                m_edgeList.emplace_back(glm::uvec2(e.a, e.b));
                m_edgeToFace.emplace_back(glm::uvec2(~0u, ~0u)); // 占位
            }
            else {
                eid = edgeIndexMap[e];
            }

            glm::uvec2& ef = m_edgeToFace[eid];
            if (ef.x == ~0u) ef.x = faceID;
            else ef.y = faceID;
        }

        for (int i = 0; i < 4; ++i) {
            uint32_t vid = vertexRemap[vs[i]];
            vertexFaces[vid].push_back(faceID);
        }
    }

    // Step 3: 建立 vertex -> edges 映射
    for (uint32_t eid = 0; eid < m_edgeList.size(); ++eid) {
        const glm::uvec2& e = m_edgeList[eid];
        uint32_t v0 = vertexRemap[e.x];
        uint32_t v1 = vertexRemap[e.y];
        vertexEdges[e.x].push_back(eid);
        vertexEdges[e.y].push_back(eid);
    }

    // face —> edges
    m_faceEdgeIndices.clear();
    m_faceEdgeIndices.reserve(m_quadFacesRaw.size());

    for (const glm::uvec4& q : m_quadFacesRaw) {
        uint32_t v[4] = { q.x, q.y, q.z, q.w };
        glm::uvec4 fe;
        for (int i = 0; i < 4; ++i) {
            Edge e(vertexRemap[v[i]],                      //  remap
                vertexRemap[v[(i + 1) & 3]]);
            fe[i] = edgeIndexMap[e];                       // 一定命中
        }
        m_faceEdgeIndices.push_back(fe);
    }

    uint32_t canonicalVertexCount = static_cast<uint32_t>(canonicalPosMap.size()); // 或用 uniqueRemapped.size()

    // Step 4: 打包 vertex->faces 索引列表
    m_vertexFaceCounts.clear();
    m_vertexFaceIndices.clear();

    for (uint32_t i = 0; i < canonicalVertexCount; ++i)
    {
        const auto& list = vertexFaces[i];        // vertexFaces 以 canonical idx 为 key
        m_vertexFaceCounts.push_back(static_cast<uint32_t>(list.size()));
        m_vertexFaceIndices.insert(m_vertexFaceIndices.end(), list.begin(), list.end());
    }


    // Step 5: 打包 vertex->edges 索引列表
    m_vertexEdgeCounts.clear();
    m_vertexEdgeIndices.clear();

    for (uint32_t i = 0; i < canonicalVertexCount; ++i)           // 也改这里
    {
        const auto& list = vertexEdges[i];
        m_vertexEdgeCounts.push_back(static_cast<uint32_t>(list.size()));
        m_vertexEdgeIndices.insert(m_vertexEdgeIndices.end(), list.begin(), list.end());
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

    weldVertices(m_vertices, m_indices);

    // convert from triangle to quad to fill m_quadFaces
    // Step 1: 提取原始 position
    std::vector<glm::vec3> rawPos;
    for (auto& v : m_vertices)
        rawPos.push_back(v.pos);

    // Step 2: 生成 quad 顶点坐标 + quad 索引
    std::vector<glm::vec3> quadPos;
    triangle_to_quads(rawPos, m_indices, quadPos, m_quadFacesRaw);

    // Step 3: 构建 m_quadVertices（带默认 normal/uv）
    m_quadVerticesRaw.resize(quadPos.size());
    for (size_t i = 0; i < quadPos.size(); ++i) {
        m_quadVerticesRaw[i].pos = quadPos[i];
        m_quadVerticesRaw[i].normal = glm::vec3(0, 1, 0); // 默认法线
        m_quadVerticesRaw[i].uv = glm::vec2(0);           // 默认 UV
    }
    generateTrianglesFromQuads();

    return true;
}

//void labutils::GltfModel::load_unit_cube()
//{
//    // ========== 原始 Quad 顶点数据 ==========
//    m_quadVerticesRaw = {
//        {{-1.f, -1.f, -1.f}, {}, {}}, // 0
//        {{ 1.f, -1.f, -1.f}, {}, {}}, // 1
//        {{ 1.f,  1.f, -1.f}, {}, {}}, // 2
//        {{-1.f,  1.f, -1.f}, {}, {}}, // 3
//        {{-1.f, -1.f,  1.f}, {}, {}}, // 4
//        {{ 1.f, -1.f,  1.f}, {}, {}}, // 5
//        {{ 1.f,  1.f,  1.f}, {}, {}}, // 6
//        {{-1.f,  1.f,  1.f}, {}, {}}, // 7
//    };
//
//    m_quadFacesRaw = {
//    {0, 1, 5, 4}, // bottom  y = -1
//    {3, 2, 6, 7}, // top     y = +1
//    {4, 5, 6, 7}, // front   z = +1
//    {0, 1, 2, 3}, // back    z = -1
//    {1, 5, 6, 2}, // right   x = +1
//    {0, 4, 7, 3}, // left    x = -1
//    };
//
//
//    // 填充 quadIndicesRaw
//    m_quadIndicesRaw.clear();
//    for (const auto& face : m_quadFacesRaw) {
//        m_quadIndicesRaw.push_back(face.x);
//        m_quadIndicesRaw.push_back(face.y);
//        m_quadIndicesRaw.push_back(face.z);
//        m_quadIndicesRaw.push_back(face.w);
//    }
//
//    // ========== 生成三角形数据（渲染用） ==========
//    m_vertices = m_quadVerticesRaw; // 顶点数据相同
//    m_indices.clear();
//
//    for (const auto& face : m_quadFacesRaw)
//    {
//        // 三角形 1: v0, v1, v2
//        m_indices.push_back(face.x);
//        m_indices.push_back(face.y);
//        m_indices.push_back(face.z);
//
//        // 三角形 2: v2, v3, v0
//        m_indices.push_back(face.z);
//        m_indices.push_back(face.w);
//        m_indices.push_back(face.x);
//    }
//}

void labutils::GltfModel::load_unit_cube()
{
    // 1) ───── 顶点坐标（8 个）
    m_vertices = {
        {{-1.f, -1.f, -1.f}, {0,1,0}, {0,0}},   // 0
        {{ 1.f, -1.f, -1.f}, {0,1,0}, {0,0}},   // 1
        {{ 1.f,  1.f, -1.f}, {0,1,0}, {0,0}},   // 2
        {{-1.f,  1.f, -1.f}, {0,1,0}, {0,0}},   // 3
        {{-1.f, -1.f,  1.f}, {0,1,0}, {0,0}},   // 4
        {{ 1.f, -1.f,  1.f}, {0,1,0}, {0,0}},   // 5
        {{ 1.f,  1.f,  1.f}, {0,1,0}, {0,0}},   // 6
        {{-1.f,  1.f,  1.f}, {0,1,0}, {0,0}},   // 7
    };

    // 2) ───── 三角形索引（每面 2 个三角 × 6 = 12）
    m_indices = {
        // +X (右)
        1,5,6,   1,6,2,
        // -X (左)
        4,0,3,   4,3,7,
        // +Y (顶)
        3,2,6,   3,6,7,
        // -Y (底)
        4,5,1,   4,1,0,
        // +Z (前)
        4,5,6,   4,6,7,
        // -Z (后)
        0,1,2,   0,2,3,
    };

    // 3) ───── 复制 glTF 路径的“3 步”
    // Step-1: 把位置提出来
    std::vector<glm::vec3> rawPos;
    rawPos.reserve(m_vertices.size());
    for (auto& v : m_vertices) rawPos.push_back(v.pos);

    // Step-2: 三角 → Quad
    std::vector<glm::vec3> quadPos;
    triangle_to_quads(rawPos, m_indices,        // 用旧算法
        quadPos, m_quadFacesRaw); // 得到 36 个 Quad

    // Step-3: 生成 Quad 顶点数组
    m_quadVerticesRaw.resize(quadPos.size());
    for (size_t i = 0; i < quadPos.size(); ++i) {
        m_quadVerticesRaw[i].pos = quadPos[i];
        m_quadVerticesRaw[i].normal = glm::vec3(0, 1, 0);
        m_quadVerticesRaw[i].uv = glm::vec2(0);
    }

    // 用现成函数把 Quad → 三角索引，供渲染
    generateTrianglesFromQuads();
}
