#define TINYGLTF_IMPLEMENTATION 
#define STB_IMAGE_IMPLEMENTATION 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#include <glm/gtc/epsilon.hpp>
#include "gltf_model.hpp"
#include <iostream>
#include <unordered_set>

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

    struct EdgeKey 
    {
        uint32_t v0, v1;

        EdgeKey(uint32_t a, uint32_t b) {
            if (a < b) { v0 = a; v1 = b; }
            else { v0 = b; v1 = a; }
        }

        bool operator==(const EdgeKey& other) const {
            return v0 == other.v0 && v1 == other.v1;
        }
    };
    struct EdgeKeyHash
    {
        std::size_t operator()(const EdgeKey& k) const {
            return std::hash<uint32_t>()(k.v0) ^ std::hash<uint32_t>()(k.v1 << 1);
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

std::vector<uint32_t> GltfModel::generateTrianglesFromQuads() const
{
    std::vector<uint32_t> out;
    out.reserve(m_quadFaces.size() * 6); // 每个 quad 2 三角形 = 6 indices

    for (const auto& q : m_quadFaces)
    {
        // Triangle 1: a, b, c
        out.push_back(q.x);
        out.push_back(q.y);
        out.push_back(q.z);

        // Triangle 2: a, c, d
        out.push_back(q.x);
        out.push_back(q.z);
        out.push_back(q.w);
    }
    return out;
}


void GltfModel::preprocessForSubdivision() {
    // 建立 position → canonical index
    std::unordered_map<glm::vec3, uint32_t, PosHasher> canonicalPosMap;
    std::vector<uint32_t> vertexRemap(m_quadVertices.size());

    for (uint32_t i = 0; i < m_quadVertices.size(); ++i) {
        glm::vec3 pos = m_quadVertices[i].pos;
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
    for (uint32_t faceID = 0; faceID < m_quadFaces.size(); ++faceID) {
        const glm::uvec4& q = m_quadFaces[faceID];
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
    m_faceEdgeIndices.reserve(m_quadFaces.size());

    for (const glm::uvec4& q : m_quadFaces) {
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


bool GltfModel::loadFromFile(const std::string& path)
{
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
    if (idxAcc >= 0) 
    {
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
    else 
    {
        m_indices.resize(vtxCount);
        std::iota(m_indices.begin(), m_indices.end(), 0u);
    }

    weldVertices(m_vertices, m_indices);

    m_quadVertices = m_vertices;

    // convert from triangle to quad to fill m_quadFaces
    // Step 1: 提取原始 position
    //std::vector<glm::vec3> rawPos;
    //for (auto& v : m_vertices)
    //    rawPos.push_back(v.pos);

    //// Step 2: 生成 quad 顶点坐标 + quad 索引
    //std::vector<glm::vec3> quadPos;
    //triangle_to_quads(rawPos, m_indices, quadPos, m_quadFaces);

    //// Step 3: 构建 m_quadVertices（带默认 normal/uv）
    //m_quadVertices.resize(quadPos.size());
    //for (size_t i = 0; i < quadPos.size(); ++i) {
    //    m_quadVertices[i].pos = quadPos[i];
    //    m_quadVertices[i].normal = glm::vec3(0, 1, 0); // 默认法线
    //    m_quadVertices[i].uv = glm::vec2(0);           // 默认 UV
    //}
    //generateTrianglesFromQuads();

    return true;
}


void GltfModel::load_unit_cube()
{
    // Vertices
    //m_vertices = {
    //    {{-1.f, -1.f, -1.f}, {0,1,0}, {0,0}},   // 0
    //    {{ 1.f, -1.f, -1.f}, {0,1,0}, {0,0}},   // 1
    //    {{ 1.f,  1.f, -1.f}, {0,1,0}, {0,0}},   // 2
    //    {{-1.f,  1.f, -1.f}, {0,1,0}, {0,0}},   // 3
    //    {{-1.f, -1.f,  1.f}, {0,1,0}, {0,0}},   // 4
    //    {{ 1.f, -1.f,  1.f}, {0,1,0}, {0,0}},   // 5
    //    {{ 1.f,  1.f,  1.f}, {0,1,0}, {0,0}},   // 6
    //    {{-1.f,  1.f,  1.f}, {0,1,0}, {0,0}},   // 7
    //};

    //// Indices
    //m_indices = {
    //    // +X
    //    1,5,6,   1,6,2,
    //    // -X
    //    4,0,3,   4,3,7,
    //    // +Y
    //    3,2,6,   3,6,7,
    //    // -Y
    //    4,5,1,   4,1,0,
    //    // +Z
    //    4,5,6,   4,6,7,
    //    // -Z
    //    0,1,2,   0,2,3,
    //};
    //m_vertices = {
    //{{-1.f, -1.f, -1.f}}, // 0
    //{{ 1.f, -1.f, -1.f}}, // 1
    //{{ 1.f,  1.f, -1.f}}, // 2
    //{{-1.f,  1.f, -1.f}}, // 3
    //{{-1.f, -1.f,  1.f}}, // 4
    //{{ 1.f, -1.f,  1.f}}, // 5
    //{{ 1.f,  1.f,  1.f}}, // 6
    //{{-1.f,  1.f,  1.f}}, // 7

    //// 面中心点（一个立方体 6 个面）
    //{{ 0.f,  0.f, -1.f}}, // 8  (-Z)
    //{{ 0.f,  0.f,  1.f}}, // 9  (+Z)
    //{{ 0.f, -1.f,  0.f}}, // 10 (-Y)
    //{{ 0.f,  1.f,  0.f}}, // 11 (+Y)
    //{{-1.f,  0.f,  0.f}}, // 12 (-X)
    //{{ 1.f,  0.f,  0.f}}, // 13 (+X)
    //};

    //m_indices = {
    //    // Each quad split into 4 triangles around its center

    //    // -Z face (0,1,2,3) and center 8
    //    0,1,8,  1,2,8,  2,3,8,  3,0,8,
    //    // +Z face (4,5,6,7) and center 9
    //    4,5,9,  5,6,9,  6,7,9,  7,4,9,
    //    // -Y face (0,1,5,4) and center 10
    //    0,1,10, 1,5,10, 5,4,10, 4,0,10,
    //    // +Y face (3,2,6,7) and center 11
    //    3,2,11, 2,6,11, 6,7,11, 7,3,11,
    //    // -X face (0,3,7,4) and center 12
    //    0,3,12, 3,7,12, 7,4,12, 4,0,12,
    //    // +X face (1,2,6,5) and center 13
    //    1,2,13, 2,6,13, 6,5,13, 5,1,13,
    //};

    m_vertices = {
    {{  1.f,  1.f,  1.f }},   // 0
    {{ -1.f, -1.f,  1.f }},   // 1
    {{ -1.f,  1.f, -1.f }},   // 2
    {{  1.f, -1.f, -1.f }},   // 3
    };
    m_indices = {
    0, 1, 2,   // Face 0
    0, 3, 1,   // Face 1
    0, 2, 3,   // Face 2
    1, 3, 2    // Face 3
    };


    m_quadVertices = m_vertices;
    m_quadIndices= m_indices;


    //// 3) ───── 复制 glTF 路径的“3 步”
    //// Step-1: 把位置提出来
    //std::vector<glm::vec3> rawPos;
    //rawPos.reserve(m_vertices.size());
    //for (auto& v : m_vertices) rawPos.push_back(v.pos);

    //// Step-2: 三角 → Quad
    //std::vector<glm::vec3> quadPos;
    //triangle_to_quads(rawPos, m_indices,        // 用旧算法
    //    quadPos, m_quadFacesRaw); // 得到 36 个 Quad

    //// Step-3: 生成 Quad 顶点数组
    //m_quadVerticesRaw.resize(quadPos.size());
    //for (size_t i = 0; i < quadPos.size(); ++i) {
    //    m_quadVerticesRaw[i].pos = quadPos[i];
    //    m_quadVerticesRaw[i].normal = glm::vec3(0, 1, 0);
    //    m_quadVerticesRaw[i].uv = glm::vec2(0);
    //}

    //generateTrianglesFromQuads();
}

void labutils::GltfModel::firstSubdivision()
{
    /* ---------- 0. 清空输出 ---------- */
    m_quadVertices.clear();
    m_quadFaces.clear();
    m_quadIndices.clear();
    m_quadLinelists.clear();
    m_edgeList.clear();
    m_edgeToFace.clear();
    m_vertexFaceCounts.clear();
    m_vertexFaceIndices.clear();
    m_vertexEdgeCounts.clear();
    m_vertexEdgeIndices.clear();
    m_faceEdgeIndices.clear();

    /* ---------- 1. 统计原三角拓扑 ---------- */
    using FaceVec = std::vector<uint32_t>;          // face id 集
    using EdgeVec = std::vector<EdgeKey>;           // incident edges

    std::unordered_map<uint32_t, FaceVec> vertexFaces;
    std::unordered_map<uint32_t, EdgeVec> vertexEdges;
    std::unordered_map<EdgeKey, FaceVec, EdgeKeyHash> edgeToFaces;

    const size_t triCount = m_indices.size() / 3;

    for (size_t t = 0; t < triCount; ++t)
    {
        uint32_t i0 = m_indices[t * 3 + 0];
        uint32_t i1 = m_indices[t * 3 + 1];
        uint32_t i2 = m_indices[t * 3 + 2];

        uint32_t faceId = static_cast<uint32_t>(t);

        EdgeKey e01(i0, i1), e12(i1, i2), e20(i2, i0);

        edgeToFaces[e01].push_back(faceId);
        edgeToFaces[e12].push_back(faceId);
        edgeToFaces[e20].push_back(faceId);

        vertexFaces[i0].push_back(faceId);
        vertexFaces[i1].push_back(faceId);
        vertexFaces[i2].push_back(faceId);

        vertexEdges[i0].push_back(e01);
        vertexEdges[i0].push_back(e20);
        vertexEdges[i1].push_back(e01);
        vertexEdges[i1].push_back(e12);
        vertexEdges[i2].push_back(e12);
        vertexEdges[i2].push_back(e20);
    }

    /* ---------- 2. 生成面点 (F) ---------- */
    std::vector<uint32_t>  facePointIndex(triCount);
    std::vector<glm::vec3> facePoints(triCount);

    for (size_t t = 0; t < triCount; ++t)
    {
        uint32_t i0 = m_indices[t * 3 + 0];
        uint32_t i1 = m_indices[t * 3 + 1];
        uint32_t i2 = m_indices[t * 3 + 2];

        glm::vec3 p = (m_vertices[i0].pos +
            m_vertices[i1].pos +
            m_vertices[i2].pos) / 3.0f;

        Vertex fv; fv.pos = p;

        facePointIndex[t] = static_cast<uint32_t>(m_quadVertices.size());
        facePoints[t] = p;
        m_quadVertices.push_back(fv);
    }

    /* ---------- 3. 生成边点 (R′) + 初步 edgeList ---------- */
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edgePointIndex;   // edge → vertex idx
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edgeIndexMap;     // edge → edgeList idx

    struct EdgeInfo { uint32_t idx, f0, f1; };
    std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> edgeMap;

    for (auto& [ekey, flist] : edgeToFaces)
    {
        /* 3.1 计算 & 存入边点 */
        glm::vec3 mid = (m_vertices[ekey.v0].pos + m_vertices[ekey.v1].pos) * 0.5f;
        glm::vec3 fAvg(0.f);
        for (uint32_t fid : flist) fAvg += facePoints[fid];
        fAvg /= float(flist.size());

        glm::vec3 edgePtPos = (mid + fAvg) * 0.5f;

        Vertex ev; ev.pos = edgePtPos;
        uint32_t vIdx = static_cast<uint32_t>(m_quadVertices.size());
        m_quadVertices.push_back(ev);
        edgePointIndex[ekey] = vIdx;

        ///* 3.2 填 edgeList & 初始 edgeMap 行 */
        //uint32_t eIdx = static_cast<uint32_t>(m_edgeList.size());
        //m_edgeList.emplace_back(ekey.v0, ekey.v1);

        //EdgeInfo info{};
        //info.idx = eIdx;
        //info.f0 = flist[0];
        //info.f1 = (flist.size() > 1) ? flist[1] : UINT32_MAX;
        //edgeIndexMap[ekey] = eIdx;
        //edgeMap[ekey] = info;
    }

    /* ---------- 4. 生成新顶点 (Q,R,S 公式) ---------- */
    std::unordered_map<uint32_t, uint32_t> newVertexIndex;  // 原顶点 id → 新顶点 idx

    for (size_t vid = 0; vid < m_vertices.size(); ++vid)
    {
        const glm::vec3 S = m_vertices[vid].pos;
        const auto& fvec = vertexFaces[vid];
        const auto& evec = vertexEdges[vid];
        const uint32_t n = static_cast<uint32_t>(fvec.size());

        /* Q = facePoints 平均 */
        glm::vec3 Q(0.f);
        for (uint32_t fid : fvec) Q += facePoints[fid];
        Q /= float(n);

        /* R = incident edge 中点平均 */
        glm::vec3 R(0.f);
        for (const auto& ek : evec)
            R += (m_vertices[ek.v0].pos + m_vertices[ek.v1].pos) * 0.5f;
        R /= float(evec.size());

        glm::vec3 newPos = (Q + 2.f * R + (float(n) - 3.f) * S) / float(n);

        Vertex nv; nv.pos = newPos;
        uint32_t newIdx = static_cast<uint32_t>(m_quadVertices.size());
        m_quadVertices.push_back(nv);
        newVertexIndex[vid] = newIdx;
    }

    /* ---------- 5. 生成 3 × quad/三角面 ---------- */
    const auto registerEdge = [&](
        uint32_t a, uint32_t b, uint32_t faceId,
        auto& edgeMapRef, auto& edgeListRef, auto& edgeIdxRef) -> uint32_t
        {
            EdgeKey key(a, b);
            auto it = edgeMapRef.find(key);
            if (it == edgeMapRef.end())
            {
                /* 新边 */
                EdgeInfo info{};
                info.idx = static_cast<uint32_t>(edgeListRef.size());
                info.f0 = faceId;
                info.f1 = UINT32_MAX;
                edgeListRef.emplace_back(key.v0, key.v1);
                edgeMapRef.emplace(key, info);
                edgeIdxRef[key] = info.idx;
                return info.idx;
            }
            else
            {
                /* 二次遇到：填 faceB */
                if (it->second.f1 == UINT32_MAX)
                    it->second.f1 = faceId;
                return it->second.idx;
            }
        };

    for (size_t t = 0; t < triCount; ++t)
    {
        uint32_t i0 = m_indices[t * 3 + 0];
        uint32_t i1 = m_indices[t * 3 + 1];
        uint32_t i2 = m_indices[t * 3 + 2];

        uint32_t v0 = newVertexIndex[i0];
        uint32_t v1 = newVertexIndex[i1];
        uint32_t v2 = newVertexIndex[i2];

        uint32_t ep01 = edgePointIndex[EdgeKey(i0, i1)];
        uint32_t ep12 = edgePointIndex[EdgeKey(i1, i2)];
        uint32_t ep20 = edgePointIndex[EdgeKey(i2, i0)];
        uint32_t fp = facePointIndex[t];

        /* 为该三角生成 3 个 quad */
        glm::uvec4 quads[3] = {
            { v0,  ep01, fp,  ep20 },
            { v1,  ep12, fp,  ep01 },
            { v2,  ep20, fp,  ep12 }
        };

        for (int q = 0; q < 3; ++q)
        {
            uint32_t quadId = static_cast<uint32_t>(m_quadFaces.size());
            m_quadFaces.push_back(quads[q]);

            /* 同时注册四条边并写 faceEdgeIndices */
            const uint32_t vs[4] = { quads[q][0], quads[q][1], quads[q][2], quads[q][3] };
            glm::uvec4 edgeIdx;
            for (int e = 0; e < 4; ++e)
            {
                uint32_t a = vs[e];
                uint32_t b = vs[(e + 1) & 3];
                edgeIdx[e] = registerEdge(a, b, quadId, edgeMap, m_edgeList, edgeIndexMap);
            }
            m_faceEdgeIndices.push_back(edgeIdx);
        }
    }

    // fill m_quadLinelists
    for (auto e : m_edgeList)
    {
        m_quadLinelists.push_back(e[0]);
        m_quadLinelists.push_back(e[1]);

    }


    /* ---------- 6. 填 m_edgeToFace (与 m_edgeList 对齐) ---------- */
    m_edgeToFace.resize(m_edgeList.size(), glm::uvec2(UINT32_MAX, UINT32_MAX));
    for (auto& [key, info] : edgeMap)
        m_edgeToFace[info.idx] = glm::uvec2(info.f0, info.f1);

    /* ---------- 7. 生成 m_quadIndices（三角化） ---------- */
    for (const auto& q : m_quadFaces)
    {
        /* quad → (0,1,2) (2,3,0) */
        m_quadIndices.push_back(q[0]);
        m_quadIndices.push_back(q[1]);
        m_quadIndices.push_back(q[2]);
        m_quadIndices.push_back(q[2]);
        m_quadIndices.push_back(q[3]);
        m_quadIndices.push_back(q[0]);
    }

    /* ---------- 8. 顶点↔面 / 顶点↔边 映射 ---------- */
    for (size_t vid = 0; vid < m_vertices.size(); ++vid)
    {
        /* face adjacency */
        const auto& fvec = vertexFaces[vid];
        m_vertexFaceCounts.push_back(static_cast<uint32_t>(fvec.size()));
        for (uint32_t fid : fvec) m_vertexFaceIndices.push_back(fid);

        /* edge adjacency */
        const auto& evec = vertexEdges[vid];
        m_vertexEdgeCounts.push_back(static_cast<uint32_t>(evec.size()));
        for (const auto& ek : evec)
            m_vertexEdgeIndices.push_back(edgeIndexMap[ek]);
    }
    //debugPrintVerticesAndIndices(m_quadVertices, m_quadIndices, "Print");
    //debugPrintEdgeList();
    //debugPrintEdgeToFace();
    //debugPrintQuadFaces();
}

void GltfModel::debugPrintVerticesAndIndices(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const std::string& name) const {
    std::cout << "=== Debug: " << name << " ===\n";
    std::cout << "Vertices (" << vertices.size() << "):\n";
    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto& v = vertices[i];
        std::cout << "  [" << i << "]: ("
            << v.pos.x << ", " << v.pos.y << ", " << v.pos.z << ")\n";
    }

    std::cout << "Indices (" << indices.size() << "):\n";
    for (size_t i = 0; i < indices.size(); i += 3) {
        std::cout << "  Triangle " << i / 3 << ": "
            << indices[i] << ", " << indices[i + 1] << ", " << indices[i + 2] << "\n";
    }
    std::cout << "============================\n";
}

void GltfModel::debugPrintEdgeList() {
    std::cout << "EdgeList (" << m_edgeList.size() << "):\n";
    for (size_t i = 0; i < m_edgeList.size(); ++i)
        std::cout << "  [" << i << "] " << m_edgeList[i][0] << " - " << m_edgeList[i][1] << "\n";
}

void GltfModel::debugPrintEdgeToFace() {
    std::cout << "EdgeToFace (" << m_edgeToFace.size() << "):\n";
    for (size_t i = 0; i < m_edgeToFace.size(); ++i)
        std::cout << "  [" << i << "] faces " << m_edgeToFace[i][0]
        << ", " << m_edgeToFace[i][1] << "\n";
}

void GltfModel::debugPrintQuadFaces() {
    std::cout << "QuadFaces (" << m_quadFaces.size() << "):\n";
    for (size_t i = 0; i < m_quadFaces.size(); ++i) {
        const auto& q = m_quadFaces[i];
        std::cout << "  [" << i << "] " << q[0] << "," << q[1]
            << "," << q[2] << "," << q[3] << "\n";
    }
}