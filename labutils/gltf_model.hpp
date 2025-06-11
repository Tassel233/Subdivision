#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>

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
		const std::vector<Vertex>& vertices() const { return m_vertices; }
		const std::vector<uint32_t>& indices() const { return m_indices; }

	private:
		std::vector<Vertex>   m_vertices;
		std::vector<uint32_t> m_indices;
	};
}
