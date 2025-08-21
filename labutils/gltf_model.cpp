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

    //struct EdgeKey 
    //{
    //    uint32_t v0, v1;

    //    EdgeKey(uint32_t a, uint32_t b) {
    //        if (a < b) { v0 = a; v1 = b; }
    //        else { v0 = b; v1 = a; }
    //    }

    //    bool operator==(const EdgeKey& other) const {
    //        return v0 == other.v0 && v1 == other.v1;
    //    }
    //};
    struct EdgeKey
    {
        uint32_t v0{}, v1{};               // 用零初始化即可

        /* 默认构造 —— 必须有！*/
        EdgeKey() = default;

        /* 带参构造：规范化顺序（a < b）*/
        EdgeKey(uint32_t a, uint32_t b)
        {
            if (a < b) { v0 = a; v1 = b; }
            else { v0 = b; v1 = a; }
        }

        bool operator==(const EdgeKey& rhs) const
        {
            return v0 == rhs.v0 && v1 == rhs.v1;
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

    static void analyseSharpAtVertex(
        uint32_t                               vId,
        const std::vector<EdgeKey>& incEdges,
        const std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash>& sharpMap,
        uint32_t& sharpCnt,
        glm::vec3                              neigh[2],
        const std::vector<Vertex>& verts)
    {
        sharpCnt = 0;
        for (auto& ek : incEdges)
        {
            auto it = sharpMap.find(ek);
            if (it != sharpMap.end() && it->second > 0)
            {
                if (sharpCnt < 2) {
                    uint32_t other = (ek.v0 == vId ? ek.v1 : ek.v0);
                    neigh[sharpCnt] = verts[other].pos;
                }
                ++sharpCnt;
            }
        }
    }

    /* edgeList + sharpness → 查表 */
    static void buildSharpMap(const std::vector<glm::uvec2>& list,
        const std::vector<uint32_t>& sharp,
        std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash>& out)
    {
        for (size_t i = 0; i < list.size(); ++i)
            out.emplace(EdgeKey(list[i][0], list[i][1]), sharp[i]);
    }
}

std::vector<uint32_t> GltfModel::generateTrianglesFromQuads() const
{
    std::vector<uint32_t> out;
    out.reserve(m_quadFaces.size() * 6); 

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
                m_edgeToFace.emplace_back(glm::uvec2(~0u, ~0u));
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
            Edge e(vertexRemap[v[i]],
                vertexRemap[v[(i + 1) & 3]]);
            fe[i] = edgeIndexMap[e];
        }
        m_faceEdgeIndices.push_back(fe);
    }

    uint32_t canonicalVertexCount = static_cast<uint32_t>(canonicalPosMap.size());

    
    m_vertexFaceCounts.clear();
    m_vertexFaceIndices.clear();

    for (uint32_t i = 0; i < canonicalVertexCount; ++i)
    {
        const auto& list = vertexFaces[i];
        m_vertexFaceCounts.push_back(static_cast<uint32_t>(list.size()));
        m_vertexFaceIndices.insert(m_vertexFaceIndices.end(), list.begin(), list.end());
    }


    
    m_vertexEdgeCounts.clear();
    m_vertexEdgeIndices.clear();

    for (uint32_t i = 0; i < canonicalVertexCount; ++i)
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

    const auto& posAccessor = model.accessors[posAcc];
    if (posAccessor.type != TINYGLTF_TYPE_VEC3 ||
        posAccessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        std::cerr << "[gltf] Invalid POSITION attribute type\n";
        return false;
    }

    int nrmAcc = getAttr("NORMAL");
    int uvAcc = getAttr("TEXCOORD_0");
    int idxAcc = prim.indices >= 0 ? prim.indices : -1;

    std::vector<uint8_t> posRaw;
    if (!readAccessor(model, posAcc, posRaw)) {
        std::cerr << "[gltf] Failed to read POSITION data\n";
        return false;
    }

    size_t vtxCount = posAccessor.count;
    m_vertices.resize(vtxCount);


    const auto* posData = reinterpret_cast<const glm::vec3*>(posRaw.data());
    for (size_t i = 0; i < vtxCount; ++i) {
        m_vertices[i].pos = posData[i];
    }

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
        for (auto& v : m_vertices) v.normal = glm::vec3(0, 1, 0);
    }




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
    m_quadIndices = m_indices;
    // just fill the data randomly
    m_quadLinelists = m_quadIndices;

    return true;
}


void GltfModel::load_unit_gemometry()
{
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


    //Tetrahedron
    //m_vertices = {
    //{{  1.f,  1.f,  1.f }},   // 0
    //{{ -1.f, -1.f,  1.f }},   // 1
    //{{ -1.f,  1.f, -1.f }},   // 2
    //{{  1.f, -1.f, -1.f }},   // 3
    //};
    //m_indices = {
    //0, 1, 2,   // Face 0
    //0, 3, 1,   // Face 1
    //0, 2, 3,   // Face 2
    //1, 3, 2    // Face 3
    //};


    //// Pyramid
    //m_vertices = {
    //{{ -1.f, 0.f, -1.f }},   
    //{{  1.f, 0.f, -1.f }},   
    //{{  1.f, 0.f,  1.f }},   
    //{{ -1.f, 0.f,  1.f }},   
    //{{  0.f, 1.f,  0.f }},   
    //};
    //m_indices = {
    //    0, 1, 4,   
    //    1, 2, 4,   
    //    2, 3, 4,   
    //    3, 0, 4,   
    //    0, 3, 2,   
    //    0, 2, 1    
    //};

    // Octahedron
    m_vertices = {
    {{  1.f,  0.f,  0.f }},   // 0
    {{ -1.f,  0.f,  0.f }},   // 1
    {{  0.f,  1.f,  0.f }},   // 2
    {{  0.f, -1.f,  0.f }},   // 3
    {{  0.f,  0.f,  1.f }},   // 4
    {{  0.f,  0.f, -1.f }}    // 5
    };

    m_indices = {
        0, 2, 4,   // Face 0
        1, 4, 2,   // Face 1
        0, 4, 3,   // Face 2
        1, 3, 4,   // Face 3
        0, 5, 2,   // Face 4
        1, 2, 5,   // Face 5
        0, 3, 5,   // Face 6
        1, 5, 3    // Face 7
    };

    initial_sharpness.resize(12, 0);
    initial_sharpness[2] = 1;
    initial_sharpness[3] = 1;
    initial_sharpness[8] = 1;
    initial_sharpness[10] = 1;




    m_quadVertices = m_vertices;
    m_quadIndices= m_indices;
    // just fill the data randomly
    m_quadLinelists = m_quadIndices;

}


void labutils::GltfModel::firstSubdivision()
{
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> sharpOld;
    {
        std::unordered_set<EdgeKey, EdgeKeyHash> seen;
        size_t sharpIdx = 0;

        const size_t triCnt = m_indices.size() / 3;
        for (size_t t = 0; t < triCnt; ++t)
        {
            uint32_t i0 = m_indices[3 * t + 0];
            uint32_t i1 = m_indices[3 * t + 1];
            uint32_t i2 = m_indices[3 * t + 2];

            EdgeKey edges[3] = { EdgeKey(i0,i1), EdgeKey(i1,i2), EdgeKey(i2,i0) };
            for (auto& ek : edges)
                if (seen.insert(ek).second)
                {
                    uint32_t s = (sharpIdx < initial_sharpness.size()) ?
                        initial_sharpness[sharpIdx] : 0;
                    sharpOld.emplace(ek, s);
                    ++sharpIdx;
                }
        }

        if (sharpIdx != initial_sharpness.size())
            std::cerr << "[firstSubdivision]  initial_sharpness counts(" << initial_sharpness.size()
            << ") don't match with(" << sharpIdx << ") \n";
    }

    
    m_quadVertices.clear();
    m_quadFaces.clear();
    m_quadIndices.clear();
    m_quadLinelists.clear();
    m_edgeList.clear();
    m_edgeToFace.clear();
    m_sharpness.clear();
    m_vertexFaceCounts.clear();
    m_vertexFaceIndices.clear();
    m_vertexEdgeCounts.clear();
    m_vertexEdgeIndices.clear();
    m_faceEdgeIndices.clear();

    using FaceVec = std::vector<uint32_t>;
    using EdgeVec = std::vector<EdgeKey>;
    std::unordered_map<uint32_t, FaceVec> vertexFaces;
    std::unordered_map<uint32_t, EdgeVec> vertexEdges;
    std::unordered_map<EdgeKey, FaceVec, EdgeKeyHash> edgeToFaces;

    const size_t triCnt = m_indices.size() / 3;
    for (size_t t = 0; t < triCnt; ++t)
    {
        uint32_t i0 = m_indices[3 * t + 0], i1 = m_indices[3 * t + 1], i2 = m_indices[3 * t + 2];
        uint32_t fid = uint32_t(t);
        EdgeKey e01(i0, i1), e12(i1, i2), e20(i2, i0);

        edgeToFaces[e01].push_back(fid);
        edgeToFaces[e12].push_back(fid);
        edgeToFaces[e20].push_back(fid);

        vertexFaces[i0].push_back(fid);  vertexFaces[i1].push_back(fid);  vertexFaces[i2].push_back(fid);
        vertexEdges[i0].push_back(e01);  vertexEdges[i0].push_back(e20);
        vertexEdges[i1].push_back(e01);  vertexEdges[i1].push_back(e12);
        vertexEdges[i2].push_back(e12);  vertexEdges[i2].push_back(e20);
    }

    
    std::vector<uint32_t>  facePointIdx(triCnt);
    std::vector<glm::vec3> facePoints(triCnt);

    for (size_t t = 0; t < triCnt; ++t)
    {
        uint32_t i0 = m_indices[3 * t + 0], i1 = m_indices[3 * t + 1], i2 = m_indices[3 * t + 2];
        glm::vec3 p = (m_vertices[i0].pos + m_vertices[i1].pos + m_vertices[i2].pos) / 3.f;

        facePointIdx[t] = uint32_t(m_quadVertices.size());
        facePoints[t] = p;
        m_quadVertices.push_back(Vertex{ p });
    }

    
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edgePtIdx;
    std::unordered_map<uint32_t, EdgeKey>             edgePtParent;

    for (auto& [ek, fl] : edgeToFaces)
    {
        glm::vec3 v0 = m_vertices[ek.v0].pos, v1 = m_vertices[ek.v1].pos;
        uint32_t  s = sharpOld[ek];

        glm::vec3 p = (s > 0) ? (v0 + v1) * 0.5f :
            ([&] { glm::vec3 f(0.f); for (uint32_t fid : fl)f += facePoints[fid];
        f /= float(fl.size()); return ((v0 + v1) * 0.5f + f) * 0.5f; }());

        uint32_t vid = uint32_t(m_quadVertices.size());
        m_quadVertices.push_back(Vertex{ p });
        edgePtIdx[ek] = vid; edgePtParent[vid] = ek;
    }

    std::unordered_map<uint32_t, uint32_t> newVIdx;

    for (uint32_t vid = 0; vid < m_vertices.size(); ++vid)
    {
        uint32_t cnt; glm::vec3 nei[2];
        analyseSharpAtVertex(vid, vertexEdges[vid], sharpOld, cnt, nei, m_vertices);

        glm::vec3 S = m_vertices[vid].pos, newPos;
        if (cnt >= 3)      newPos = S;
        else if (cnt == 2) newPos = (nei[0] + 6.f * S + nei[1]) / 8.f;
        else            /* smooth */ {
            glm::vec3 Q(0.f); for (uint32_t fid : vertexFaces[vid]) Q += facePoints[fid];
            Q /= float(vertexFaces[vid].size());
            glm::vec3 R(0.f); for (auto& ek : vertexEdges[vid]) R += (m_vertices[ek.v0].pos + m_vertices[ek.v1].pos) * 0.5f;
            R /= float(vertexEdges[vid].size());
            newPos = (Q + 2.f * R + (float(vertexFaces[vid].size()) - 3.f) * S) / float(vertexFaces[vid].size());
        }

        uint32_t nid = uint32_t(m_quadVertices.size());
        m_quadVertices.push_back(Vertex{ newPos });
        newVIdx[vid] = nid;
    }

    struct EdgeInfo { uint32_t idx, f0, f1; };
    std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> edgeMap;
    //std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edgeIdxMap;
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edgeIndexMap;

    auto regEdge = [&](uint32_t a, uint32_t b, uint32_t sharp, uint32_t fid)->uint32_t
        {
            EdgeKey k(a, b);
            auto it = edgeMap.find(k);
            if (it == edgeMap.end()) {
                EdgeInfo inf{ uint32_t(m_edgeList.size()),fid,UINT32_MAX };
                edgeMap.emplace(k, inf); edgeIndexMap[k] = inf.idx;
                m_edgeList.emplace_back(k.v0, k.v1);
                m_sharpness.push_back(sharp);
                return inf.idx;
            }
            else { if (it->second.f1 == UINT32_MAX) it->second.f1 = fid; return it->second.idx; }
        };

    for (size_t t = 0; t < triCnt; ++t)
    {
        uint32_t i0 = m_indices[3 * t + 0], i1 = m_indices[3 * t + 1], i2 = m_indices[3 * t + 2];
        uint32_t v0 = newVIdx[i0], v1 = newVIdx[i1], v2 = newVIdx[i2];
        uint32_t e01 = edgePtIdx[EdgeKey(i0, i1)], e12 = edgePtIdx[EdgeKey(i1, i2)], e20 = edgePtIdx[EdgeKey(i2, i0)];
        uint32_t fp = facePointIdx[t];

        uint32_t s01 = sharpOld[EdgeKey(i0, i1)], s12 = sharpOld[EdgeKey(i1, i2)], s20 = sharpOld[EdgeKey(i2, i0)];

        glm::uvec4 quads[3] = { {v0,e01,fp,e20},{v1,e12,fp,e01},{v2,e20,fp,e12} };

        for (int q = 0; q < 3; ++q)
        {
            uint32_t fNew = uint32_t(m_quadFaces.size());
            m_quadFaces.push_back(quads[q]);

            const uint32_t vv[4] = { quads[q][0],quads[q][1],quads[q][2],quads[q][3] };
            glm::uvec4 eIdx;
            for (int e = 0; e < 4; ++e)
            {
                uint32_t a = vv[e], b = vv[(e + 1) & 3];

                uint32_t sharp = 0;
                auto childOf = [&](EdgeKey parent)->bool {
                    return (edgePtParent.count(a) && edgePtParent[a] == parent) ||
                        (edgePtParent.count(b) && edgePtParent[b] == parent);
                    };
                if (childOf(EdgeKey(i0, i1))) sharp = s01 ? s01 - 1 : 0;
                else if (childOf(EdgeKey(i1, i2))) sharp = s12 ? s12 - 1 : 0;
                else if (childOf(EdgeKey(i2, i0))) sharp = s20 ? s20 - 1 : 0;

                eIdx[e] = regEdge(a, b, sharp, fNew);
            }
            m_faceEdgeIndices.push_back(eIdx);
        }
    }

    m_edgeToFace.resize(m_edgeList.size(), glm::uvec2(UINT32_MAX, UINT32_MAX));
    for (auto& [key, info] : edgeMap)
        m_edgeToFace[info.idx] = glm::uvec2(info.f0, info.f1);

    for (const auto& q : m_quadFaces)
    {
        m_quadIndices.push_back(q[0]);
        m_quadIndices.push_back(q[1]);
        m_quadIndices.push_back(q[2]);
        m_quadIndices.push_back(q[2]);
        m_quadIndices.push_back(q[3]);
        m_quadIndices.push_back(q[0]);
    }

    //for (size_t vid = 0; vid < m_vertices.size(); ++vid)
    //{
    //    /* face adjacency */
    //    const auto& fvec = vertexFaces[vid];
    //    m_vertexFaceCounts.push_back(static_cast<uint32_t>(fvec.size()));
    //    for (uint32_t fid : fvec) m_vertexFaceIndices.push_back(fid);

    //    /* edge adjacency */
    //    const auto& evec = vertexEdges[vid];
    //    m_vertexEdgeCounts.push_back(static_cast<uint32_t>(evec.size()));
    //    for (const auto& ek : evec)
    //        m_vertexEdgeIndices.push_back(edgeIndexMap[ek]);
    //}
    // 

    const uint32_t Vp = static_cast<uint32_t>(m_quadVertices.size());

    std::vector<std::vector<uint32_t>> vFaces(Vp);   
    std::vector<std::vector<uint32_t>> vEdges(Vp);   

    for (uint32_t fid = 0; fid < m_quadFaces.size(); ++fid)
    {
        const glm::uvec4& q = m_quadFaces[fid];


        for (int k = 0; k < 4; ++k)
            vFaces[q[k]].push_back(fid);

        const uint32_t v[4] = { q[0], q[1], q[2], q[3] };
        for (int e = 0; e < 4; ++e)
        {
            uint32_t a = v[e];
            uint32_t b = v[(e + 1) & 3];

            EdgeKey key(a, b);                 
            uint32_t eid = edgeIndexMap[key];  

            vEdges[a].push_back(eid);
            vEdges[b].push_back(eid);         
        }
    }

    m_vertexFaceCounts.reserve(Vp);
    m_vertexEdgeCounts.reserve(Vp);
    m_vertexFaceIndices.reserve(Vp * 4);              
    m_vertexEdgeIndices.reserve(m_edgeList.size() * 2);

    for (uint32_t vid = 0; vid < Vp; ++vid)
    {
        m_vertexFaceCounts.push_back(static_cast<uint32_t>(vFaces[vid].size()));
        for (uint32_t fid : vFaces[vid])
            m_vertexFaceIndices.push_back(fid);

        m_vertexEdgeCounts.push_back(static_cast<uint32_t>(vEdges[vid].size()));
        for (uint32_t eid : vEdges[vid])
            m_vertexEdgeIndices.push_back(eid);
    }
    //debugPrintVerticesAndIndices(m_quadVertices, m_quadIndices, "Print");
    //debugPrintEdgeList();
    //debugPrintEdgeToFace();
    //debugPrintQuadFaces();
    m_quadLinelists.clear();
    m_quadLinelists.reserve(m_edgeList.size() * 2); 

    for (const auto& edge : m_edgeList) {
        m_quadLinelists.push_back(edge.x); 
        m_quadLinelists.push_back(edge.y); 
    }
}

void labutils::GltfModel::subdivideQuadOnce()
{

    const auto oldVerts = m_quadVertices;
    const auto oldFaces = m_quadFaces;
    const auto oldEdges = m_edgeList;
    const auto oldSharp = m_sharpness;
    const size_t faceCnt = oldFaces.size();

    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> sharpOld;
    buildSharpMap(oldEdges, oldSharp, sharpOld);

    m_quadVertices.clear();   m_quadFaces.clear();  m_quadIndices.clear();
    m_quadLinelists.clear();  m_edgeList.clear();   m_edgeToFace.clear();
    m_sharpness.clear();      m_vertexFaceCounts.clear();
    m_vertexFaceIndices.clear(); m_vertexEdgeCounts.clear();
    m_vertexEdgeIndices.clear(); m_faceEdgeIndices.clear();

    using FaceVec = std::vector<uint32_t>;
    using EdgeVec = std::vector<EdgeKey>;
    std::unordered_map<uint32_t, FaceVec> vertexFaces;
    std::unordered_map<uint32_t, EdgeVec> vertexEdges;
    std::unordered_map<EdgeKey, FaceVec, EdgeKeyHash> edgeToFaces;

    for (uint32_t fid = 0; fid < faceCnt; ++fid)
    {
        const glm::uvec4& q = oldFaces[fid];
        uint32_t v[4] = { q[0],q[1],q[2],q[3] };
        EdgeKey e01(v[0], v[1]), e12(v[1], v[2]), e23(v[2], v[3]), e30(v[3], v[0]);

        edgeToFaces[e01].push_back(fid); edgeToFaces[e12].push_back(fid);
        edgeToFaces[e23].push_back(fid); edgeToFaces[e30].push_back(fid);

        for (int i = 0; i < 4; ++i) {
            vertexFaces[v[i]].push_back(fid);
            vertexEdges[v[i]].push_back(EdgeKey(v[i], v[(i + 1) & 3]));
        }
    }

    std::vector<uint32_t> facePtIdx(faceCnt);
    std::vector<glm::vec3> facePts(faceCnt);
    for (uint32_t fid = 0; fid < faceCnt; ++fid)
    {
        auto& q = oldFaces[fid];
        glm::vec3 p = (oldVerts[q[0]].pos + oldVerts[q[1]].pos +
            oldVerts[q[2]].pos + oldVerts[q[3]].pos) * 0.25f;
        facePtIdx[fid] = uint32_t(m_quadVertices.size());
        facePts[fid] = p;
        m_quadVertices.push_back(Vertex{ p });
    }

    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edgePtIdx;
    std::unordered_map<uint32_t, EdgeKey>             edgePtParent;

    for (auto& [ek, fl] : edgeToFaces)
    {
        glm::vec3 v0 = oldVerts[ek.v0].pos, v1 = oldVerts[ek.v1].pos;
        uint32_t  s = sharpOld[ek];
        glm::vec3 p = (s > 0) ? (v0 + v1) * 0.5f :
            ([&] {glm::vec3 f(0.f); for (uint32_t fid : fl)f += facePts[fid];
        f /= float(fl.size()); return ((v0 + v1) * 0.5f + f) * 0.5f; }());

        uint32_t vid = uint32_t(m_quadVertices.size());
        m_quadVertices.push_back(Vertex{ p });
        edgePtIdx[ek] = vid; edgePtParent[vid] = ek;
    }

    std::unordered_map<uint32_t, uint32_t> newVIdx;
    for (uint32_t vid = 0; vid < oldVerts.size(); ++vid)
    {
        uint32_t cnt; glm::vec3 nei[2];
        analyseSharpAtVertex(vid, vertexEdges[vid], sharpOld, cnt, nei, oldVerts);

        glm::vec3 S = oldVerts[vid].pos, newPos;
        if (cnt >= 3)      newPos = S;
        else if (cnt == 2) newPos = (nei[0] + 6.f * S + nei[1]) / 8.f;
        else {/* smooth */
            glm::vec3 Q(0.f); for (uint32_t fid : vertexFaces[vid]) Q += facePts[fid];
            Q /= float(vertexFaces[vid].size());
            glm::vec3 R(0.f); for (auto& ek : vertexEdges[vid]) R += (oldVerts[ek.v0].pos + oldVerts[ek.v1].pos) * 0.5f;
            R /= float(vertexEdges[vid].size());
            newPos = (Q + 2.f * R + (float(vertexFaces[vid].size()) - 3.f) * S) / float(vertexFaces[vid].size());
        }

        uint32_t nid = uint32_t(m_quadVertices.size());
        m_quadVertices.push_back(Vertex{ newPos });
        newVIdx[vid] = nid;
    }

    struct EdgeInfo { uint32_t idx, f0, f1; };
    std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> edgeMap;
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edgeIndexMap;

    auto regEdge = [&](uint32_t a, uint32_t b, uint32_t sharp, uint32_t fid)->uint32_t
        {
            EdgeKey k(a, b);
            auto it = edgeMap.find(k);
            if (it == edgeMap.end()) {
                EdgeInfo inf{ uint32_t(m_edgeList.size()),fid,UINT32_MAX };
                edgeMap.emplace(k, inf); edgeIndexMap[k] = inf.idx;
                m_edgeList.emplace_back(k.v0, k.v1);
                m_sharpness.push_back(sharp);
                return inf.idx;
            }
            else { if (it->second.f1 == UINT32_MAX) it->second.f1 = fid; return it->second.idx; }
        };

    auto childSharp = [&](uint32_t ep, uint32_t other)->uint32_t
        {
            auto it = edgePtParent.find(ep);
            if (it == edgePtParent.end()) return 0;
            uint32_t s = sharpOld[it->second];
            if (s == 0) return 0;
            return s - 1;                     
        };

    for (uint32_t fid = 0; fid < faceCnt; ++fid)
    {
        auto& q = oldFaces[fid];
        uint32_t v0 = newVIdx[q[0]], v1 = newVIdx[q[1]],
            v2 = newVIdx[q[2]], v3 = newVIdx[q[3]];
        uint32_t e01 = edgePtIdx[EdgeKey(q[0], q[1])],
            e12 = edgePtIdx[EdgeKey(q[1], q[2])],
            e23 = edgePtIdx[EdgeKey(q[2], q[3])],
            e30 = edgePtIdx[EdgeKey(q[3], q[0])],
            fp = facePtIdx[fid];

        glm::uvec4 quads[4] = { {v0,e01,fp,e30},{v1,e12,fp,e01},
                             {v2,e23,fp,e12},{v3,e30,fp,e23} };

        for (int qd = 0; qd < 4; ++qd)
        {
            uint32_t fNew = uint32_t(m_quadFaces.size());
            m_quadFaces.push_back(quads[qd]);

            const uint32_t vv[4] = { quads[qd][0],quads[qd][1],quads[qd][2],quads[qd][3] };
            glm::uvec4 eIdx;
            for (int e = 0; e < 4; ++e)
            {
                uint32_t a = vv[e], b = vv[(e + 1) & 3];
                uint32_t sharp = childSharp(a, b);
                eIdx[e] = regEdge(a, b, sharp, fNew);
            }
            m_faceEdgeIndices.push_back(eIdx);
        }
    }
    for (auto e : m_edgeList)
    {
        m_quadLinelists.push_back(e[0]);
        m_quadLinelists.push_back(e[1]);
    }

    m_edgeToFace.resize(m_edgeList.size(), glm::uvec2(UINT32_MAX, UINT32_MAX));
    for (auto& [key, info] : edgeMap)
        m_edgeToFace[info.idx] = glm::uvec2(info.f0, info.f1);

    for (const auto& q : m_quadFaces)
    {
        m_quadIndices.push_back(q[0]); m_quadIndices.push_back(q[1]); m_quadIndices.push_back(q[2]);
        m_quadIndices.push_back(q[2]); m_quadIndices.push_back(q[3]); m_quadIndices.push_back(q[0]);
    }

    const uint32_t Vp = static_cast<uint32_t>(m_quadVertices.size());
    std::vector<std::vector<uint32_t>> vFaces(Vp);
    std::vector<std::vector<uint32_t>> vEdges(Vp);

    for (uint32_t fid = 0; fid < m_quadFaces.size(); ++fid)
    {
        const glm::uvec4& q = m_quadFaces[fid];
        for (int k = 0; k < 4; ++k)
            vFaces[q[k]].push_back(fid);

        const uint32_t v[4] = { q[0], q[1], q[2], q[3] };
        for (int e = 0; e < 4; ++e)
        {
            EdgeKey key(v[e], v[(e + 1) & 3]);
            uint32_t eid = edgeIndexMap[key];

            vEdges[v[e]].push_back(eid);
            vEdges[v[(e + 1) & 3]].push_back(eid);
        }
    }

    m_vertexFaceCounts.reserve(Vp);
    m_vertexEdgeCounts.reserve(Vp);
    m_vertexFaceIndices.reserve(Vp * 4);
    m_vertexEdgeIndices.reserve(m_edgeList.size() * 2);

    for (uint32_t vid = 0; vid < Vp; ++vid)
    {
        m_vertexFaceCounts.push_back(static_cast<uint32_t>(vFaces[vid].size()));
        for (uint32_t fid : vFaces[vid]) m_vertexFaceIndices.push_back(fid);

        m_vertexEdgeCounts.push_back(static_cast<uint32_t>(vEdges[vid].size()));
        for (uint32_t eid : vEdges[vid]) m_vertexEdgeIndices.push_back(eid);
    }
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