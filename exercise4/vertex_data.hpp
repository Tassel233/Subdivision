#pragma once

#include <cstdint>

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
#include "../labutils/gltf_model.hpp"


struct ColorizedMesh
{
	labutils::Buffer positions;
	labutils::Buffer colors;

	std::uint32_t vertexCount;
};

struct ModelMesh
{
	labutils::Buffer posBuffer;
	labutils::Buffer indexBuffer;

	std::uint32_t indicesCount;
};


ColorizedMesh create_triangle_mesh( labutils::VulkanContext const&, labutils::Allocator const& );

ColorizedMesh create_plane_mesh( labutils::VulkanContext const&, labutils::Allocator const& );

ModelMesh create_model_mesh(labutils::VulkanContext const&, labutils::Allocator const&, labutils::GltfModel const&);