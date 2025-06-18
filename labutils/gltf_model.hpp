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


	class GltfModel
	{
	public:
		bool loadFromFile(const std::string& path);
		const std::vector<Vertex>& get_vertices() const { return m_vertices; }
		const std::vector<uint32_t>& get_indices() const { return m_indices; }

		const std::vector<Vertex>& get_quad_vertices() const { return m_quadVertices; }
		const std::vector<uint32_t>& get_quad_indices() const { return m_quadIndices; }

		const uint32_t& get_vertexCount() const { return m_vertices.size(); }
		const uint32_t& get_quad_vertexCount() const { return m_quadVertices.size(); }

		void generateTrianglesFromQuads();


	private:
		std::vector<Vertex>   m_vertices;
		std::vector<uint32_t> m_indices;

		std::vector<Vertex> m_quadVertices;
		std::vector<glm::uvec4> m_quadFaces;
		std::vector<uint32_t> m_quadIndices;
	};

}