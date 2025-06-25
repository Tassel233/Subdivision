#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>



namespace labutils
{
	struct Vertex {
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec2 uv;
	};

	// λ�� hash
	struct PosHasher {
		size_t operator()(glm::vec3 const& p) const noexcept {
			size_t h1 = std::hash<float>{}(p.x);
			size_t h2 = std::hash<float>{}(p.y);
			size_t h3 = std::hash<float>{}(p.z);
			return h1 ^ (h2 << 1) ^ (h3 << 2);
		}
	};

	inline bool pos_equal(glm::vec3 const& a,
		glm::vec3 const& b,
		float eps = 1e-6f)
	{
		return glm::length(a - b) < eps;
	}

	class GltfModel
	{
	public:

		bool loadFromFile(const std::string& path);
		const std::vector<Vertex>& get_vertices() const { return m_vertices; }
		const std::vector<uint32_t>& get_indices() const { return m_indices; }

		const std::vector<Vertex>& get_quad_vertices() const { return m_quadVerticesRaw; }
		const std::vector<uint32_t>& get_quad_indices() const { return m_quadIndicesRaw; }
		const std::vector<glm::uvec4>& get_quad_faces() const { return m_quadFacesRaw;  }

		const uint32_t& get_vertexCount() const { return m_vertices.size(); }
		const uint32_t& get_quad_vertexCount() const { return m_quadVerticesRaw.size(); }

		void generateTrianglesFromQuads();
		void preprocessForSubdivision();
		void load_unit_cube();

		// raw triangle data
		std::vector<Vertex>   m_vertices;
		std::vector<uint32_t> m_indices;

		// raw quad data
		std::vector<Vertex> m_quadVerticesRaw;
		std::vector<glm::uvec4> m_quadFacesRaw;
		std::vector<uint32_t> m_quadIndicesRaw;

		// transferred to buffer and used in shader
		std::vector<glm::uvec2> m_edgeList;
		std::vector<glm::uvec2> m_edgeToFace;
		std::vector<uint32_t> m_vertexFaceCounts;
		std::vector<uint32_t> m_vertexFaceIndices;
		std::vector<uint32_t> m_vertexEdgeCounts;
		std::vector<uint32_t> m_vertexEdgeIndices;
		std::vector<glm::uvec4> m_faceEdgeIndices;


	private:





	};

}