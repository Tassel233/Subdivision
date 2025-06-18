workspace "COMP5892M-vulkan"
	language "C++"
	cppdialect "C++20"

	platforms { "x64" }
	configurations { "debug", "release" }

	flags "NoPCH"
	flags "MultiProcessorCompile"

	startproject "exercise4"

	debugdir "%{wks.location}"
	objdir "_build_/%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	targetsuffix "-%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	
	-- Default toolset options
	filter "toolset:gcc or toolset:clang"
		linkoptions { "-pthread" }
		buildoptions { "-march=native", "-Wall", "-pthread" }

	filter "toolset:msc-*"
		defines { "_CRT_SECURE_NO_WARNINGS=1" }
		defines { "_SCL_SECURE_NO_WARNINGS=1" }
		buildoptions { "/utf-8" }
	
	filter "*"

	-- default options for GLSLC
	glslcOptions = "-O --target-env=vulkan1.2"

	-- default libraries
	filter "system:linux"
		links "dl"
	
	filter "system:windows"

	filter "*"

	-- default outputs
	filter "kind:StaticLib"
		targetdir "lib/"

	filter "kind:ConsoleApp"
		targetdir "bin/"
		targetextension ".exe"
	
	filter "*"

	--configurations
	filter "debug"
		symbols "On"
		defines { "_DEBUG=1" }

	filter "release"
		optimize "On"
		defines { "NDEBUG=1" }

	filter "*"

	defines { "SOLUTION_CODE=1" }
	glslcOptions = glslcOptions .. " -DSOLUTION_CODE"

-- Third party dependencies
include "third_party" 

-- GLSLC helpers
dofile( "util/glslc.lua" )

-- Projects





project "exercise4"
	local sources = { 
		"exercise4/**.cpp",
		"exercise4/**.hpp",
		"exercise4/**.hxx"
	}

	kind "ConsoleApp"
	location "exercise4"

	files( sources )


	dependson "exercise4-shaders"

	links "labutils"
	links "x-volk"
	links "x-stb"
	links "x-glfw"
	links "x-vma"
	-- links "x-tinygltf"

	dependson "x-glm" 

project "exercise4-shaders"
	local shaders = { 
		"exercise4/shaders/*.vert",
		"exercise4/shaders/*.frag",
		"exercise4/shaders/*.comp"
	}

	kind "Utility"
	location "exercise4/shaders"

	files( shaders )

	handle_glsl_files( glslcOptions, "assets/exercise4/shaders", {} )


project "labutils"
	local sources = { 
		"labutils/**.cpp",
		"labutils/**.hpp",
		"labutils/**.hxx"
	}

	kind "StaticLib"
	location "labutils"

	files( sources )

	--tinygltf
	includedirs { "third_party/tinygltf" }
    files {
        "third_party/tinygltf/tiny_gltf.h",
        "third_party/tinygltf/json.hpp",
        "third_party/tinygltf/stb_image.h",
        "third_party/tinygltf/stb_image_write.h"
    }

project()

--EOF
