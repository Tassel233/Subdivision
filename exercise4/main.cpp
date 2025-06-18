#include <iostream>
#include <chrono>
#include <volk/volk.h>

#include <tuple>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "vertex_data.hpp"

#include "../labutils/gltf_model.hpp"

namespace
{
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
		// See sources in exercise4/shaders/*. 
#		define MODELDIR_ "assets/exercise4/"
#		define SHADERDIR_ "assets/exercise4/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "shader3d.vert.spv";
		constexpr char const* kVertModelPath = SHADERDIR_ "shadermodel.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "shader3d.frag.spv";
		constexpr char const* kFragModelPath = SHADERDIR_ "shadermodel.frag.spv";
		constexpr char const* kFragWirePath = SHADERDIR_ "wireframe.frag.spv";
		constexpr char const* kCompShaderPath = SHADERDIR_ "test.comp.spv";




#		undef SHADERDIR_
		constexpr char const* modelPath = MODELDIR_ "scene.gltf";
		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;



		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.03f;
		constexpr float kCameraFar   = 10.f;

		constexpr auto kCameraFov    = 60.0_degf;

		// General rule: for debugging, you want to be able to move around quickly in the scene (but slow down if ecessary).
		// The exact settings here depend on the scene scale and similar settings.
		constexpr float kCameraBaseSpeed = 0.8f; // units/second
		constexpr float kCameraFastMult = 5.f; // speed multiplier
		constexpr float kCameraSlowMult = 0.05f; // speed multiplier
		constexpr float kCameraMouseSensitivity = 0.005f; // radians per pixel
	}
	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_button(GLFWwindow*, int, int, int);
	// receive mouse position
	void glfw_callback_motion(GLFWwindow*, double, double);
	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		max,
	};

	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.0f, mouseY = 0.0f;
		float previousX = 0.0f, previousY = 0.0f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};

	// update state based on elapsed time
	void update_user_state(UserState&, float aElapsedTime);

	// Uniform data
	namespace glsl
	{
		struct SceneUniform
		{
			// Note: need to be careful about the packing/alignment here!
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
		};

		// We want to use vkCmdUpdateBuffer() to update the contents of our uniform buffers.
		// vkCmdUpdateBuffer() has a number of requirements, including the two below. See:
		// https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdUpdateBuffer.html
		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");


	}

	// Helpers:
	lut::RenderPass create_render_pass( lut::VulkanWindow const& );

	lut::DescriptorSetLayout create_scene_descriptor_layout( lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_object_descriptor_layout( lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_compute_descriptor_layout(lut::VulkanWindow const&);

	lut::PipelineLayout create_pipeline_layout( lut::VulkanContext const&, VkDescriptorSetLayout );
	lut::Pipeline create_pipeline( lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout );
	lut::Pipeline create_model_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_wireframe_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);
	lut::Pipeline create_compute_pipeline(lut::VulkanWindow const&, VkPipelineLayout);




	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);


	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState&
	);


	void record_commands(
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkPipeline,
		VkExtent2D const&,
		VkBuffer aPositionBuffer,
		VkBuffer aColorBuffer,
		std::uint32_t aVertexCount,
		VkBuffer aSceneUBO,
		glsl::SceneUniform const&,
		VkPipelineLayout,
		VkDescriptorSet aSceneDescriptors
	);

	void record_commands1(
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkPipeline,
		VkPipeline,
		VkExtent2D const&,
		VkBuffer aPositionBuffer,
		VkBuffer aIndexBuffer,
		std::uint32_t aIndicesCount,
		VkBuffer aSceneUBO,
		glsl::SceneUniform const&,
		VkPipelineLayout,
		VkDescriptorSet aSceneDescriptors
	);

	void record_compute_commands(
		VkCommandBuffer aCmdBuff,
		VkPipeline aComputePipeline,
		VkPipelineLayout aPipelineLayout,
		VkDescriptorSet aComputeDescriptors,
		std::uint32_t workGroupCountX // = ceil(numElements / local_size_x
	);



	void submit_commands(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
	void present_results( 
		VkQueue, 
		VkSwapchainKHR, 
		std::uint32_t aImageIndex, 
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);
}


int main() try
{
	labutils::GltfModel model;
	if (model.loadFromFile(cfg::modelPath))
	{
		std::cout << "load successfully!" << std::endl;
		//uploadMesh(model.vertices(), model.indices());
	}
	else
	{
		std::cout << "load failed!" << std::endl;
	}


	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	// Configure the GLFW window
	//glfwSetKeyCallback( window.window, &glfw_callback_key_press );

	// Configure the GLFW window
	UserState state{};
	glfwSetWindowUserPointer(window.window, &state);

	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);


	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator( window );

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass( window );

	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout( window );
	//TODO- (Section 4) create object descriptor set layout


	lut::PipelineLayout pipeLayout = create_pipeline_layout( window, sceneLayout.handle);
	//lut::Pipeline pipe = create_pipeline( window, renderPass.handle, pipeLayout.handle );
	lut::Pipeline pipe = create_model_pipeline( window, renderPass.handle, pipeLayout.handle);
	lut::Pipeline wire_pipe = create_wireframe_pipeline(window, renderPass.handle, pipeLayout.handle);


	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

	lut::CommandPool cpool = lut::create_command_pool( window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );

	std::size_t frameIndex = 0;
	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> frameDone;
	std::vector<lut::Semaphore> imageAvailable, renderFinished;
	
	for( std::size_t i = 0; i < framebuffers.size(); ++i )
	{
		cbuffers.emplace_back( lut::alloc_command_buffer( window, cpool.handle ) );
		frameDone.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
		imageAvailable.emplace_back( lut::create_semaphore( window ) );
		renderFinished.emplace_back( lut::create_semaphore( window ) );
	}

	// Load data
	//ColorizedMesh planeMesh = create_plane_mesh( window, allocator );
	ModelMesh modelMesh= create_model_mesh(window, allocator, model);

	// Create scene uniform buffer with lut::create_buffer()
	lut::Buffer sceneUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		0,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
	);

	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);


	// Allocate descriptor set for uniform buffer
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(
		window,
		dpool.handle,
		sceneLayout.handle
	);

	// Initialize descriptor set with vkUpdateDescriptorSets

	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}


	//TODO- (Section 4) load textures into image
	//TODO- (Section 4) create image view for texture image
	//TODO- (Section 4) create default texture sampler
	//TODO- (Section 4) allocate and initialize descriptor sets for texture

	//record current time
	auto previousClock = Clock_::now();

	// Application main loop
	bool recreateSwapchain = false;

	//////////////////////////////////////////////////////////////////////////////////////////////////

	// compute shader test
	lut::Buffer dataBuffer = create_buffer(
		allocator,
		10 * sizeof(int),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		0,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);
	// wirte in data
	int* mapped = nullptr;
	vmaMapMemory(allocator.allocator, dataBuffer.allocation, (void**)&mapped);
	for (int i = 0; i < 10; ++i) mapped[i] = i;
	vmaUnmapMemory(allocator.allocator, dataBuffer.allocation);
	// set descriptorlayout
	lut::DescriptorSetLayout computeLayout = create_compute_descriptor_layout(window);
	lut::PipelineLayout computepipeLayout = create_pipeline_layout(window, computeLayout.handle);

	VkDescriptorSet SSBODescriptors = lut::alloc_desc_set(
		window,
		dpool.handle,
		computeLayout.handle
	);

	lut::Pipeline compPipe = create_compute_pipeline(window, computepipeLayout.handle);
	// bind descriptor
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo SSBOInfo{};
		SSBOInfo.buffer = dataBuffer.buffer;
		SSBOInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = SSBODescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

		//desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &SSBOInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}




	VkCommandBuffer compbuffer = lut::alloc_command_buffer(window, cpool.handle);
	vmaMapMemory(allocator.allocator, dataBuffer.allocation, (void**)&mapped);
	for (int i = 0; i < 10; ++i)
		printf("Result[%d] = %d\n", i, mapped[i]);
	vmaUnmapMemory(allocator.allocator, dataBuffer.allocation);

	// record commands
	record_compute_commands(compbuffer, compPipe.handle, computepipeLayout.handle, SSBODescriptors, 10);



	// submit commands
		// 1. 创建 fence，用于同步 GPU 完成信号
	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

	VkFence fence = VK_NULL_HANDLE;
	if (vkCreateFence(window.device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create compute fence");
	}

	// 2. 设置提交信息
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &compbuffer;

	// 3. 提交到 compute queue
	if (vkQueueSubmit(window.graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
		throw std::runtime_error("Failed to submit compute commands");
	}

	// 4. 等待 GPU 执行完成（阻塞）
	vkWaitForFences(window.device, 1, &fence, VK_TRUE, UINT64_MAX);

	// 5. 清理 fence
	vkDestroyFence(window.device, fence, nullptr);


	// 9. 回读并验证结果
	vmaMapMemory(allocator.allocator, dataBuffer.allocation, (void**)&mapped);
	for (int i = 0; i < 10; ++i)
		printf("Result[%d] = %d\n", i, mapped[i]);
	vmaUnmapMemory(allocator.allocator, dataBuffer.allocation);
	///////////////////////////////////////////////////////////////////////////////////////////

	while( !glfwWindowShouldClose( window.window ) )
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if( recreateSwapchain )
		{
			// We need to destroy several objects, which may still be in use by the GPU. Therefore, first wait for the GPU
			// to finish processing.
			vkDeviceWaitIdle(window.device);

			// Recreate them
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);

			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				//pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				pipe = create_model_pipeline(window, renderPass.handle, pipeLayout.handle);
				wire_pipe = create_wireframe_pipeline(window, renderPass.handle, pipeLayout.handle);
				//alpha_pipe = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);
			}

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			recreateSwapchain = false;
		}

		// Advance to next frame
		frameIndex++;
		//printf("cbuffers.size(): %d", cbuffers.size());
		frameIndex %= cbuffers.size();

		// Make sure that the frame resources are no longer in use
		assert(frameIndex < frameDone.size());

		if (auto const res = vkWaitForFences(window.device, 1, &frameDone[frameIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for frame fence %u\n"
				"vkWaitForFences() returned %s", frameIndex, lut::to_string(res).c_str());
		}

		// Acquire next swap chain image
		assert(frameIndex < imageAvailable.size());

		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable[frameIndex].handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			// This occurs e.g., when the window has been resized. In this case we need to recreate the swap chain to
			// match the new dimensions. Any resources that directly depend on the swap chain need to be recreated
			// as well. While rare, re-creating the swap chain may give us a different image format, which we should
			// handle appropriately.
			//
			// In both cases, we set the flag that the swap chain has to be re-created and jump to the top of the loop.
			// Technically, with the VK_SUBOPTIMAL_KHR return code, we could continue rendering with the
			// current swap chain (unlike VK_ERROR_OUT_OF_DATE_KHR, which does require us to recreate the
			// swap chain).
			recreateSwapchain = true;

			// We won't render a frame this time around. Consequently, no commands were submitted for execution
			// and the associated fence won't be signalled. Stepping back one frame avoids this problem.
			--frameIndex;
			frameIndex %= cbuffers.size();

			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire next swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}

		// Reset fence
		// Do this only after AcquireNextImage(), so that we can wait on the same fence again in case the swapchain
		// had to be re-created.
		if (auto const res = vkResetFences(window.device, 1, &frameDone[frameIndex].handle);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset frame fence %u\n"
				"vkResetFences() returned %s", frameIndex, lut::to_string(res).c_str());
		}

		assert(std::size_t(frameIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		// Update state
		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt);

		// Prepare data for this frame
		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);



		// Record and submit commands for this frame
		//record_commands(
		//	cbuffers[frameIndex],
		//	renderPass.handle,
		//	framebuffers[imageIndex].handle,
		//	pipe.handle,
		//	window.swapchainExtent,
		//	planeMesh.positions.buffer,
		//	planeMesh.colors.buffer,
		//	planeMesh.vertexCount,
		//	sceneUBO.buffer,
		//	sceneUniforms,
		//	pipeLayout.handle,
		//	sceneDescriptors
		//);

		record_commands1(
			cbuffers[frameIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			pipe.handle,
			wire_pipe.handle,
			window.swapchainExtent,
			modelMesh.posBuffer.buffer,
			modelMesh.indexBuffer.buffer,
			modelMesh.indicesCount,
			sceneUBO.buffer,
			sceneUniforms,
			pipeLayout.handle,
			sceneDescriptors
		);


		assert(std::size_t(frameIndex) < renderFinished.size());

		submit_commands(
			window,
			cbuffers[frameIndex],
			frameDone[frameIndex].handle,
			imageAvailable[frameIndex].handle,
			renderFinished[frameIndex].handle
		);

		present_results(
			window.presentQueue,
			window.swapchain,
			imageIndex,
			renderFinished[frameIndex].handle,
			recreateSwapchain
		);
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle( window.device );

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	void glfw_callback_key_press( GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/ )
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}

		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;

		default:;
		}

	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}


}

namespace
{
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState& aState)
	{
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.f; // mirror Y axis

		aSceneUniforms.camera = glm::inverse(aState.camera2world);

		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			// Only update the rotation on the second frame of mouse navigation.
			// This ensures that the previousX and Y variables are initialized to sensible values.
			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.0f, 0.0f, 0.0f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.0f, 1.0f, 0.0f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}

		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.0f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.0f);

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.0f, 0.0f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.0f, 0.0f, move));

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.0f, 0.0f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(move, 0.0f, 0.0f));

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.0f, move, 0.0f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.0f, -move, 0.0f));
	}

}

namespace
{
	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; // this refers to attachments[0]
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// New:
		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1; // this refers to attachments[1]
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		// New line for depth attachment
		subpasses[0].pDepthStencilAttachment = &depthAttachment;


		// Requires a subpass dependency to ensure that the first transition happens after the presentation engine is done with it.
		// https://github.com/KhronosGroup/Vulkan-Docs/wiki/Synchronization-Examples-(Legacy-synchronization-APIs)#combined-graphicspresent-queue
		// WARNING: The following has changed! Make sure to update it!
		VkSubpassDependency deps[2]{};
		deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[0].srcAccessMask = 0;
		deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		deps[0].dstSubpass = 0;
		deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		deps[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		deps[1].dstSubpass = 0;
		deps[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		deps[1].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;


		// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkRenderPassCreateInfo.html
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 2; // different dependency, same code
		passInfo.pDependencies = deps; // different dependency, same code

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n"
				"vkCreateRenderPass() returned %s\n", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);

	}

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout)
	{
		VkDescriptorSetLayout layouts[] = {
			// Order must match the set = N in the shaders
			aSceneLayout // set 0
		};

		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]); // updated!
		layoutInfo.pSetLayouts = layouts; // updated!
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s",
				lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);

	}

	lut::Pipeline create_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{

		//Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

		// Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};

		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkPipelineVertexInputStateCreateInfo inputInfo{};

		// declares how data is read from buffers
		VkVertexInputBindingDescription vertexInputs[2]{};
		vertexInputs[0].binding = 0; // bind the corresponding input buffer when drawing
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		// how the data is mapped to shaders
		VkVertexInputAttributeDescription vertexAttributes[2]{};

		// Position attribute
		vertexAttributes[0].binding = 0; // must match binding above
		vertexAttributes[0].location = 0; // must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		// index attribute
		vertexAttributes[1].binding = 1; // must match binding above
		vertexAttributes[1].location = 1; // must match shader
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;


		inputInfo.vertexBindingDescriptionCount = 2; // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = vertexInputs;

		inputInfo.vertexAttributeDescriptionCount = 2; // number of vertexAttributes above
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		// Define which primitive (point, line, triangle, ...) the input is assembled into for rasterization.
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;

		viewport.width = aWindow.swapchainExtent.width;
		viewport.height = aWindow.swapchainExtent.height;

		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // Required.


		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// We define one blend state per color attachment - this example uses a single color attachment, so we only need one.
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;
			

		// Create pipeline
		// finally!
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states
		pipeInfo.pDepthStencilState = &depthInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_model_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{

		//Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertModelPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragModelPath);

		// Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};

		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		// Pull data from the vertex buffer
		VkPipelineVertexInputStateCreateInfo inputInfo{};

		// Declare how data is read from buffer
		VkVertexInputBindingDescription vertexInputs[1]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(glm::vec3);
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//vertexInputs[1].binding = 1;
		//vertexInputs[1].stride = sizeof(float) * 2;
		//vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		// Map data to vertex shaders' input
		VkVertexInputAttributeDescription vertexAttributes[1]{};

		// Position attribute
		vertexAttributes[0].binding = 0; // must match binding above
		vertexAttributes[0].location = 0; // must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		// Index attribute
		//vertexAttributes[1].binding = 1; // must match binding above
		//vertexAttributes[1].location = 1; // must match shader
		//vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		//vertexAttributes[1].offset = 0;


		inputInfo.vertexBindingDescriptionCount = 1; // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = vertexInputs;

		inputInfo.vertexAttributeDescriptionCount = 1; // number of vertexAttributes above
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		// Define which primitive (point, line, triangle, ...) the input is assembled into for rasterization.
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;

		viewport.width = aWindow.swapchainExtent.width;
		viewport.height = aWindow.swapchainExtent.height;

		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // Required.


		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// We define one blend state per color attachment - this example uses a single color attachment, so we only need one.
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;


		// Create pipeline
		// finally!
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		//pipeInfo.pDepthStencilState = nullptr; // no depth or stencil buffers
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states
		pipeInfo.pDepthStencilState = &depthInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);

	}

	lut::Pipeline create_wireframe_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		//Load shader modules
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertModelPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragWirePath);

		// Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[2]{};

		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		// Pull data from the vertex buffer
		VkPipelineVertexInputStateCreateInfo inputInfo{};

		// Declare how data is read from buffer
		VkVertexInputBindingDescription vertexInputs[1]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(glm::vec3);
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		//vertexInputs[1].binding = 1;
		//vertexInputs[1].stride = sizeof(float) * 2;
		//vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		// Map data to vertex shaders' input
		VkVertexInputAttributeDescription vertexAttributes[1]{};

		// Position attribute
		vertexAttributes[0].binding = 0; // must match binding above
		vertexAttributes[0].location = 0; // must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		// Index attribute
		//vertexAttributes[1].binding = 1; // must match binding above
		//vertexAttributes[1].location = 1; // must match shader
		//vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		//vertexAttributes[1].offset = 0;


		inputInfo.vertexBindingDescriptionCount = 1; // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = vertexInputs;

		inputInfo.vertexAttributeDescriptionCount = 1; // number of vertexAttributes above
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		// Define which primitive (point, line, triangle, ...) the input is assembled into for rasterization.
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;

		viewport.width = aWindow.swapchainExtent.width;
		viewport.height = aWindow.swapchainExtent.height;

		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization(draw lines!!)
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_LINE;
		rasterInfo.cullMode = VK_CULL_MODE_NONE;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.f; // Required.
		// Set bias for depth-buffer
		rasterInfo.depthBiasEnable = VK_TRUE;
		rasterInfo.depthBiasConstantFactor = -0.1f;   // translate a little for wireframes before mesh
		rasterInfo.depthBiasSlopeFactor = -1.0f;


		// Define multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state
		// We define one blend state per color attachment - this example uses a single color attachment, so we only need one.
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;


		// Create pipeline
		// finally!
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		//pipeInfo.pDepthStencilState = nullptr; // no depth or stencil buffers
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states
		pipeInfo.pDepthStencilState = &depthInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);

	}

	lut::Pipeline create_compute_pipeline(lut::VulkanWindow const& aWindow, VkPipelineLayout aPipelineLayout)
	{

		//Load shader modules
		lut::ShaderModule comp = lut::load_shader_module(aWindow, cfg::kCompShaderPath);
		//lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragModelPath);

		// Define shader stages in the pipeline
		VkPipelineShaderStageCreateInfo stages[1]{};

		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_COMPUTE_BIT;
		stages[0].module = comp.handle;
		stages[0].pName = "main";

		
		VkComputePipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipeInfo.stage = stages[0];
		pipeInfo.layout = aPipelineLayout;

		
		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateComputePipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create compute pipeline\n"
				"vkCreateComputePipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);

	}







	void create_swapchain_framebuffers(
		lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass,
		std::vector<lut::Framebuffer>& aFramebuffers,
		VkImageView aDepthView
	)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] = {
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; // normal framebuffer
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n"
					"vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());
			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

	}

	lut::DescriptorSetLayout create_scene_descriptor_layout( lut::VulkanWindow const& aWindow )
	{
		// Step 1: Describe binding for the uniform buffer
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // must match binding = 0 in the shader
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		// Step 2: Fill layout create info
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		// Step 3: Create the descriptor set layout
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s",
				lut::to_string(res).c_str());
		}

		// Step 4: Return wrapped descriptor set layout
		return lut::DescriptorSetLayout(aWindow.device, layout);

	}
	lut::DescriptorSetLayout create_object_descriptor_layout( lut::VulkanWindow const& aWindow )
	{
		throw lut::Error( "Not yet implemented" ); //TODO: (Section 4) implement me!
	}
	lut::DescriptorSetLayout create_compute_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		// Step 1: Describe binding for the storage buffer
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // must match binding = 0 in the shader
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		// Step 2: Fill layout create info
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		// Step 3: Create the descriptor set layout
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s",
				lut::to_string(res).c_str());
		}

		// Step 4: Return wrapped descriptor set layout
		return lut::DescriptorSetLayout(aWindow.device, layout);

	}


	void record_commands(
		VkCommandBuffer aCmdBuff,
		VkRenderPass aRenderPass,
		VkFramebuffer aFramebuffer,
		VkPipeline aGraphicsPipe,
		VkExtent2D const& aImageExtent,
		VkBuffer aPositionBuffer,
		VkBuffer aColorBuffer,
		std::uint32_t aVertexCount,
		VkBuffer aSceneUBO,
		glsl::SceneUniform const& aSceneUniform,
		VkPipelineLayout aGraphicsLayout,
		VkDescriptorSet aSceneDescriptors
	)
	{
		// Begin recording commands
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res) {
			throw lut::Error(
				"Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}



		// Upload scene uniforms
		lut::buffer_barrier(
			aCmdBuff,
			aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(
			aCmdBuff,
			aSceneUBO,
			0,
			sizeof(glsl::SceneUniform),
			&aSceneUniform
		);

		lut::buffer_barrier(
			aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);



		// Begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background.
		clearValues[0].color.float32[1] = 0.1f; // Helps identify render pass visually
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.0f;

		clearValues[1].depthStencil.depth = 1.f; // new!


		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width, aImageExtent.height };
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Begin drawing with our graphics pipeline

		// Bind pipeline and descriptors
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);


		// Bind vertex input
		VkBuffer buffers[2] = { aPositionBuffer, aColorBuffer };
		VkDeviceSize offsets[2]{};
		vkCmdBindVertexBuffers(aCmdBuff, 0, 2, buffers, offsets);

		// Draw vertices
		vkCmdDraw(aCmdBuff, aVertexCount, 1, 0, 0);

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);


		// End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res) {
			throw lut::Error(
				"Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s",
				lut::to_string(res).c_str()
			);
		}
	}

	void record_commands1(
		VkCommandBuffer aCmdBuff,
		VkRenderPass aRenderPass,
		VkFramebuffer aFramebuffer,
		VkPipeline aGraphicsPipe,
		VkPipeline aWireframePipe,
		VkExtent2D const& aImageExtent,
		VkBuffer aPositionBuffer,
		VkBuffer aIndexBuffer,
		std::uint32_t aIndicesCount,
		VkBuffer aSceneUBO,
		glsl::SceneUniform const& aSceneUniform,
		VkPipelineLayout aGraphicsLayout,
		VkDescriptorSet aSceneDescriptors
	)
	{
		// Begin recording commands
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res) {
			throw lut::Error(
				"Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}



		// Upload scene uniforms
		lut::buffer_barrier(
			aCmdBuff,
			aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(
			aCmdBuff,
			aSceneUBO,
			0,
			sizeof(glsl::SceneUniform),
			&aSceneUniform
		);

		lut::buffer_barrier(
			aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);



		// Begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background.
		clearValues[0].color.float32[1] = 0.1f; // Helps identify render pass visually
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.0f;

		clearValues[1].depthStencil.depth = 1.f; // new!


		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = VkExtent2D{ aImageExtent.width, aImageExtent.height };
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Begin drawing with our graphics pipeline

		// Bind for mesh fill
		// Bind pipeline and descriptors
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
		// Bind buffers
		VkDeviceSize posOffset = 0;
		vkCmdBindVertexBuffers(aCmdBuff, 0, 1, &aPositionBuffer, &posOffset);
		vkCmdBindIndexBuffer(aCmdBuff, aIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
		// Draw indexed meshes
		vkCmdDrawIndexed(aCmdBuff, aIndicesCount, 1, 0, 0, 0);

		// Binding for wireframes
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aWireframePipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
		vkCmdBindVertexBuffers(aCmdBuff, 0, 1, &aPositionBuffer, &posOffset);
		vkCmdBindIndexBuffer(aCmdBuff, aIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(aCmdBuff, aIndicesCount, 1, 0, 0, 0);



		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);


		// End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res) {
			throw lut::Error(
				"Unable to end recording command buffer\n"
				"vkEndCommandBuffer() returned %s",
				lut::to_string(res).c_str()
			);
		}

	}

	void record_compute_commands(
		VkCommandBuffer aCmdBuff,
		VkPipeline aComputePipeline,
		VkPipelineLayout aPipelineLayout,
		VkDescriptorSet aComputeDescriptors,
		std::uint32_t workGroupCountX // = ceil(numElements / local_size_x
	)
	{
		// 1. Begin recording
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res) {
			throw lut::Error(
				"Unable to begin recording compute command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}

		// 2. Bind compute pipeline
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, aComputePipeline);

		// 3. Bind descriptor sets (e.g. SSBO)
		vkCmdBindDescriptorSets(
			aCmdBuff,
			VK_PIPELINE_BIND_POINT_COMPUTE,
			aPipelineLayout,
			0, 1,
			&aComputeDescriptors,
			0, nullptr
		);

		// 4. Dispatch compute shader
		vkCmdDispatch(aCmdBuff, workGroupCountX, 1, 1);

		// 5. End recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res) {
			throw lut::Error(
				"Unable to end compute command buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}
	}



	void submit_commands(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages =
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}

	}

	void submit_compute_commands(lut::VulkanWindow const& aWindow, VkCommandBuffer cmdBuf)
	{
		// 1. 创建 fence，用于同步 GPU 完成信号
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

		VkFence fence = VK_NULL_HANDLE;
		if (vkCreateFence(aWindow.device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create compute fence");
		}

		// 2. 设置提交信息
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuf;

		// 3. 提交到 compute queue
		if (vkQueueSubmit(aWindow.presentQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("Failed to submit compute commands");
		}

		// 4. 等待 GPU 执行完成（阻塞）
		vkWaitForFences(aWindow.device, 1, &fence, VK_TRUE, UINT64_MAX);

		// 5. 清理 fence
		vkDestroyFence(aWindow.device, fence, nullptr);
	}

	void present_results( VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain )
	{
		// Present the results
		// https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPresentInfoKHR.html
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}

	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		// Create the image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1,
			0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(depthImage), lut::ImageView(aWindow.device, view) };

	}

}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
